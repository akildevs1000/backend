"""
FILE: backend/app/attendance_service.py
Business logic for recording and querying attendance
"""

import logging
from datetime import datetime, date, timedelta, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import select, and_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import Attendance, Employee, AuditLog
from app.face_matching import MatchResult

logger = logging.getLogger(__name__)


class AttendanceService:

    async def clock_in(
        self,
        match_result: MatchResult,
        company_id: str,
        db: AsyncSession,
        gps_location: Optional[dict] = None,
        device_id: Optional[str] = None,
        photo_path: Optional[str] = None,
    ) -> dict:
        """
        Record a clock-in event for a verified employee.
        - Prevents duplicate clock-ins within DUPLICATE_CHECK_MINUTES
        - If employee already clocked in today, returns existing record
        """
        if not match_result.matched:
            return {
                "success": False,
                "action": None,
                "reason": match_result.reason,
                "similarity_score": match_result.similarity_score,
            }

        employee_id = match_result.employee_id
        now = datetime.now(timezone.utc)
        today = now.date()

        # Check for existing attendance record today
        existing = await self._get_today_attendance(employee_id, db)

        if existing:
            if existing.clock_out is not None:
                # Already clocked out — this is an error or end of day re-scan
                return {
                    "success": False,
                    "action": "already_clocked_out",
                    "reason": "You have already clocked out for today.",
                    "clock_in": existing.clock_in.isoformat(),
                    "clock_out": existing.clock_out.isoformat(),
                    "employee_name": match_result.employee_name,
                    "similarity_score": match_result.similarity_score,
                }

            # Already clocked in — prevent duplicate within window
            minutes_since_clock_in = (now - existing.clock_in).total_seconds() / 60
            if minutes_since_clock_in < settings.DUPLICATE_CHECK_MINUTES:
                return {
                    "success": True,
                    "action": "already_clocked_in",
                    "attendance_id": str(existing.attendance_id),
                    "clock_in": existing.clock_in.isoformat(),
                    "employee_name": match_result.employee_name,
                    "department": match_result.department,
                    "similarity_score": match_result.similarity_score,
                    "message": f"Already clocked in at {existing.clock_in.strftime('%H:%M')}",
                }

        # Create new attendance record
        attendance = Attendance(
            attendance_id=uuid4(),
            employee_id=employee_id,
            company_id=company_id,
            date=datetime(today.year, today.month, today.day, tzinfo=timezone.utc),
            clock_in=now,
            gps_location=gps_location,
            verification_method="face_recognition",
            similarity_score=match_result.similarity_score,
            liveness_score=match_result.liveness_score,
            device_id=device_id,
            clock_in_photo=photo_path,
            status=self._determine_status(now),
        )

        db.add(attendance)
        await db.flush()

        await self._log_audit(
            db=db,
            company_id=company_id,
            employee_id=employee_id,
            action="clock_in",
            details={
                "attendance_id": str(attendance.attendance_id),
                "similarity_score": match_result.similarity_score,
                "liveness_score": match_result.liveness_score,
            },
            device_id=device_id,
        )

        logger.info(f"Clock-in recorded: employee={employee_id} time={now.isoformat()}")

        return {
            "success": True,
            "action": "clock_in",
            "attendance_id": str(attendance.attendance_id),
            "clock_in": now.isoformat(),
            "employee_name": match_result.employee_name,
            "department": match_result.department,
            "similarity_score": match_result.similarity_score,
            "status": attendance.status,
            "message": f"Clock In Successful — Welcome, {match_result.employee_name}!",
        }

    async def clock_out(
        self,
        match_result: MatchResult,
        company_id: str,
        db: AsyncSession,
        gps_location: Optional[dict] = None,
        device_id: Optional[str] = None,
        photo_path: Optional[str] = None,
    ) -> dict:
        """Record clock-out for an employee who has already clocked in today."""
        if not match_result.matched:
            return {
                "success": False,
                "action": None,
                "reason": match_result.reason,
            }

        employee_id = match_result.employee_id
        now = datetime.now(timezone.utc)

        existing = await self._get_today_attendance(employee_id, db)

        if not existing:
            return {
                "success": False,
                "action": "not_clocked_in",
                "reason": "No clock-in found for today. Please clock in first.",
                "employee_name": match_result.employee_name,
            }

        if existing.clock_out is not None:
            return {
                "success": False,
                "action": "already_clocked_out",
                "reason": "You have already clocked out for today.",
                "clock_in": existing.clock_in.isoformat(),
                "clock_out": existing.clock_out.isoformat(),
                "employee_name": match_result.employee_name,
            }

        # Calculate work duration
        duration_minutes = int((now - existing.clock_in).total_seconds() / 60)

        existing.clock_out = now
        existing.clock_out_photo = photo_path
        existing.work_duration_minutes = duration_minutes
        if gps_location:
            existing.gps_location = {**(existing.gps_location or {}), "clock_out": gps_location}

        await db.flush()

        await self._log_audit(
            db=db,
            company_id=company_id,
            employee_id=employee_id,
            action="clock_out",
            details={
                "attendance_id": str(existing.attendance_id),
                "work_duration_minutes": duration_minutes,
            },
            device_id=device_id,
        )

        hours = duration_minutes // 60
        mins = duration_minutes % 60
        logger.info(f"Clock-out recorded: employee={employee_id} duration={duration_minutes}m")

        return {
            "success": True,
            "action": "clock_out",
            "attendance_id": str(existing.attendance_id),
            "clock_in": existing.clock_in.isoformat(),
            "clock_out": now.isoformat(),
            "work_duration_minutes": duration_minutes,
            "employee_name": match_result.employee_name,
            "department": match_result.department,
            "similarity_score": match_result.similarity_score,
            "message": f"Clock Out Successful — Goodbye, {match_result.employee_name}! "
                       f"You worked {hours}h {mins}m today.",
        }

    async def get_attendance_history(
        self,
        employee_id: Optional[str],
        company_id: str,
        db: AsyncSession,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        page: int = 1,
        page_size: int = 30,
    ) -> dict:
        """Fetch paginated attendance records."""
        query = select(Attendance).where(Attendance.company_id == company_id)

        if employee_id:
            query = query.where(Attendance.employee_id == employee_id)

        if start_date:
            query = query.where(Attendance.date >= datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc))

        if end_date:
            end_dt = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc)
            query = query.where(Attendance.date <= end_dt)

        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Paginate
        query = query.order_by(desc(Attendance.date))
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await db.execute(query)
        records = result.scalars().all()

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "records": [self._serialize_attendance(r) for r in records],
        }

    async def get_daily_summary(
        self, company_id: str, target_date: date, db: AsyncSession
    ) -> dict:
        """Get attendance summary for a specific day."""
        start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)

        result = await db.execute(
            select(Attendance).where(
                Attendance.company_id == company_id,
                Attendance.date >= start,
                Attendance.date < end,
            )
        )
        records = result.scalars().all()

        present = [r for r in records if r.status == "present"]
        late = [r for r in records if r.status == "late"]
        clocked_out = [r for r in records if r.clock_out is not None]

        return {
            "date": target_date.isoformat(),
            "total_present": len(records),
            "on_time": len(present),
            "late": len(late),
            "clocked_out": len(clocked_out),
            "still_clocked_in": len(records) - len(clocked_out),
            "avg_work_duration_minutes": (
                sum(r.work_duration_minutes or 0 for r in clocked_out) / len(clocked_out)
                if clocked_out else 0
            ),
        }

    # ─── Helpers ────────────────────────────────────────────────────────────────

    async def _get_today_attendance(self, employee_id: str, db: AsyncSession) -> Optional[Attendance]:
        today = datetime.now(timezone.utc).date()
        start = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)

        result = await db.execute(
            select(Attendance).where(
                Attendance.employee_id == employee_id,
                Attendance.date >= start,
                Attendance.date < end,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _determine_status(clock_in_time: datetime) -> str:
        """Determine if employee is on time or late (9:30 AM cutoff)."""
        local_hour = clock_in_time.hour
        local_minute = clock_in_time.minute
        # Standard check: late if after 9:30 AM UTC (adjust for your timezone)
        if local_hour > 9 or (local_hour == 9 and local_minute > 30):
            return "late"
        return "present"

    @staticmethod
    def _serialize_attendance(record: Attendance) -> dict:
        return {
            "attendance_id": str(record.attendance_id),
            "employee_id": str(record.employee_id),
            "date": record.date.date().isoformat() if record.date else None,
            "clock_in": record.clock_in.isoformat() if record.clock_in else None,
            "clock_out": record.clock_out.isoformat() if record.clock_out else None,
            "work_duration_minutes": record.work_duration_minutes,
            "status": record.status,
            "gps_location": record.gps_location,
            "similarity_score": record.similarity_score,
            "liveness_score": record.liveness_score,
        }

    @staticmethod
    async def _log_audit(
        db: AsyncSession,
        company_id: str,
        employee_id: str,
        action: str,
        details: dict,
        device_id: Optional[str] = None,
    ):
        log = AuditLog(
            company_id=company_id,
            employee_id=employee_id,
            action=action,
            details=details,
            device_id=device_id,
            success=True,
        )
        db.add(log)


attendance_service = AttendanceService()
