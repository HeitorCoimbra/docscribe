from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..core.pdf import generate_pdf_report
from ..schemas import coerce_confirmed_leitos
from ..database import get_db
from ..models import User
from ..repositories.thread import ThreadRepository

router = APIRouter(tags=["pdf"])


@router.get("/threads/{thread_id}/pdf")
def download_pdf(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate and download a PDF of all confirmed leitos for a thread."""
    thread_repo = ThreadRepository(db)
    thread = thread_repo.get_thread(thread_id)
    if not thread or thread.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Thread not found")

    confirmed_leitos: dict = coerce_confirmed_leitos(dict(thread.confirmed_leitos or {}))
    summaries = list(confirmed_leitos.values())

    if not summaries:
        raise HTTPException(status_code=404, detail="No confirmed leitos to export")

    try:
        pdf_bytes = generate_pdf_report(summaries)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}") from exc

    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"sumarios_plantao_{date_str}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
