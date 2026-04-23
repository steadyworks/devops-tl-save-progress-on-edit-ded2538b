from datetime import datetime
from typing import Optional, Self
from uuid import UUID

from fastapi import HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.dal import (
    DALPages,
    DALPhotobookComments,
    DALPhotobooks,
    DALPhotobookSettings,
    DAOPagesUpdate,
    DAOPhotobookCommentsCreate,
    DAOPhotobookCommentsUpdate,
    DAOPhotobooksCreate,
    DAOPhotobookSettingsCreate,
    DAOPhotobooksUpdate,
    FilterOp,
    OrderDirection,
    safe_commit,
    safe_transaction,
)
from backend.db.data_models import (
    CommentStatus,
    DAOPhotobooks,
    FontStyle,
    NotificationStatus,
    PhotobookStatus,
    UserProvidedOccasion,
)
from backend.db.externals import (
    PhotobookCommentsOverviewResponse,
    PhotobooksOverviewResponse,
)
from backend.lib.asset_manager.base import AssetManager
from backend.lib.utils.common import utcnow
from backend.route_handler.base import RouteHandler
from backend.worker.job_processor.types import JobType, PhotobookGenerationInputPayload

from .base import enforce_response_model, unauthenticated_route
from .page import PagesFullResponse


class UploadedFileInfo(BaseModel):
    filename: str
    storage_key: str


class FailedUploadInfo(BaseModel):
    filename: str
    error: str


class NewPhotobookRequest(BaseModel):
    user_provided_occasion: UserProvidedOccasion
    user_provided_custom_details: Optional[str] = None
    user_provided_context: Optional[str] = None
    asset_ids: list[UUID]


class NewPhotobookResponse(BaseModel):
    photobook_id: UUID


class PhotobookEditTitleRequest(BaseModel):
    new_title: str


class PhotobooksFullResponse(PhotobooksOverviewResponse):
    pages: list[PagesFullResponse]
    comments: list[PhotobookCommentsOverviewResponse]

    @classmethod
    async def rendered_from_dao(
        cls: type[Self],
        dao: DAOPhotobooks,
        db_session: AsyncSession,
        asset_manager: AssetManager,
    ) -> Self:
        resp = await PhotobooksOverviewResponse.rendered_from_dao(
            dao,
            db_session,
            asset_manager,
        )
        pages = await DALPages.list_all(
            db_session,
            {"photobook_id": (FilterOp.EQ, dao.id)},
            order_by=[("page_number", OrderDirection.ASC)],
        )
        pages_response_full = await PagesFullResponse.rendered_from_daos(
            pages, db_session, asset_manager
        )
        comments = await DALPhotobookComments.list_all(
            db_session,
            filters={
                "photobook_id": (FilterOp.EQ, dao.id),
                "status": (FilterOp.EQ, CommentStatus.VISIBLE),
            },
            order_by=[("created_at", OrderDirection.DESC)],
        )
        comments_response = [
            PhotobookCommentsOverviewResponse.from_dao(c) for c in comments
        ]

        return cls(
            **resp.model_dump(),
            pages=pages_response_full,
            comments=comments_response,
        )


class EditPageRequest(BaseModel):
    page_id: UUID
    new_user_message: str


class PhotobookEditPagesRequest(BaseModel):
    edits: list[EditPageRequest]


class PhotobookDeleteResponse(BaseModel):
    success: bool
    error_message: Optional[str] = None


class CreateCommentRequest(BaseModel):
    body: str


class CreateCommentResponse(BaseModel):
    comment: PhotobookCommentsOverviewResponse


class EditCommentRequest(BaseModel):
    body: str


class EditCommentResponse(BaseModel):
    comment: PhotobookCommentsOverviewResponse


class PhotobookAPIHandler(RouteHandler):
    def register_routes(self) -> None:
        self.route("/api/photobook/new", "photobook_new", ["POST"])
        self.route("/api/photobook/{photobook_id}", "get_photobook_by_id", ["GET"])
        self.route(
            "/api/photobook/{photobook_id}/edit_title", "photobook_edit_title", ["POST"]
        )
        self.route(
            "/api/photobook/{photobook_id}/edit_pages", "photobook_edit_pages", ["POST"]
        )
        self.route("/api/photobook/{photobook_id}/delete", "photobook_delete", ["POST"])
        self.route(
            "/api/photobook/{photobook_id}/comment",
            "photobook_create_comment",
            ["POST"],
        )
        self.route(
            "/api/photobook/{photobook_id}/comment/{comment_id}/edit",
            "photobook_edit_comment",
            ["POST"],
        )

    @enforce_response_model
    async def photobook_new(
        self,
        request: Request,
        payload: NewPhotobookRequest,
    ) -> NewPhotobookResponse:
        request_context = await self.get_request_context(request)

        # Persist metadata with new photobook DB entry
        async with self.app.new_db_session() as db_session:
            async with safe_transaction(
                db_session, context="photobook creation DB write", raise_on_fail=True
            ):
                photobook = await DALPhotobooks.create(
                    db_session,
                    DAOPhotobooksCreate(
                        user_id=request_context.user_id,
                        title=f"New Photobook {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        caption=None,
                        theme=None,
                        status=PhotobookStatus.PENDING,
                        user_provided_occasion=payload.user_provided_occasion,
                        user_provided_occasion_custom_details=payload.user_provided_custom_details,
                        user_provided_context=payload.user_provided_context,
                        thumbnail_asset_id=None,
                        deleted_at=None,
                        status_last_edited_by=None,
                    ),
                )
                await DALPhotobookSettings.create(
                    db_session,
                    DAOPhotobookSettingsCreate(
                        photobook_id=photobook.id,
                        main_style=None,
                        font=FontStyle.UNSPECIFIED,
                        is_comment_enabled=False,
                        is_allow_download_all_images_enabled=False,
                        is_tipping_enabled=False,
                    ),
                )

            # Enqueue photobook generation job
            await self.app.remote_job_manager_cpu_bound.enqueue(
                JobType.REMOTE_PHOTOBOOK_GENERATION,
                job_payload=PhotobookGenerationInputPayload(
                    user_id=request_context.user_id,
                    originating_photobook_id=photobook.id,
                    asset_ids=payload.asset_ids,
                ),
                max_retries=2,
                db_session=db_session,
            )

        return NewPhotobookResponse(
            photobook_id=photobook.id,
        )

    @unauthenticated_route
    @enforce_response_model
    async def get_photobook_by_id(
        self,
        photobook_id: UUID,
    ) -> PhotobooksFullResponse:
        async with self.app.new_db_session() as db_session:
            # Step 1: Fetch photobook
            photobook = await DALPhotobooks.get_by_id(db_session, photobook_id)
            if photobook is None:
                raise HTTPException(status_code=404, detail="Photobook not found")
            return await PhotobooksFullResponse.rendered_from_dao(
                photobook, db_session, self.app.asset_manager
            )

    @enforce_response_model
    async def photobook_edit_title(
        self, photobook_id: UUID, payload: PhotobookEditTitleRequest
    ) -> PhotobooksOverviewResponse:
        async with self.app.new_db_session() as db_session:
            async with safe_commit(db_session):
                photobook = await DALPhotobooks.update_by_id(
                    db_session,
                    photobook_id,
                    DAOPhotobooksUpdate(
                        title=payload.new_title,
                    ),
                )
            return await PhotobooksOverviewResponse.rendered_from_dao(
                photobook, db_session, self.app.asset_manager
            )

    @enforce_response_model
    async def photobook_edit_pages(
        self, photobook_id: UUID, payload: PhotobookEditPagesRequest
    ) -> PhotobooksFullResponse:
        async with self.app.new_db_session() as db_session:
            # 1. Validate photobook exists
            photobook = await DALPhotobooks.get_by_id(db_session, photobook_id)
            if photobook is None:
                raise HTTPException(status_code=404, detail="Photobook not found")

            # 2. Batch apply page updates
            async with safe_commit(db_session):
                update_map = {
                    edit.page_id: DAOPagesUpdate(user_message=edit.new_user_message)
                    for edit in payload.edits
                }
                await DALPages.update_many_by_ids(db_session, update_map)

            # 3. Return updated photobook and its pages
            return await PhotobooksFullResponse.rendered_from_dao(
                photobook, db_session, self.app.asset_manager
            )

    @enforce_response_model
    async def photobook_delete(
        self,
        photobook_id: UUID,
    ) -> PhotobookDeleteResponse:
        async with self.app.new_db_session() as db_session:
            photobook = await DALPhotobooks.get_by_id(db_session, photobook_id)
            if photobook is None:
                return PhotobookDeleteResponse(
                    success=False, error_message="Photobook not found"
                )

            if (
                photobook.status == PhotobookStatus.DELETED
                or photobook.status == PhotobookStatus.PERMANENTLY_DELETED
            ):
                return PhotobookDeleteResponse(
                    success=False, error_message="Photobook already deleted"
                )

            async with safe_commit(db_session):
                await DALPhotobooks.update_by_id(
                    db_session,
                    photobook_id,
                    DAOPhotobooksUpdate(
                        deleted_at=utcnow(), status=PhotobookStatus.DELETED
                    ),
                )

            return PhotobookDeleteResponse(success=True)

    @enforce_response_model
    async def photobook_create_comment(
        self,
        request: Request,
        photobook_id: UUID,
        payload: CreateCommentRequest,
    ) -> CreateCommentResponse:
        request_context = await self.get_request_context(request)

        async with self.app.new_db_session() as db_session:
            async with safe_commit(db_session):
                comment = await DALPhotobookComments.create(
                    db_session,
                    DAOPhotobookCommentsCreate(
                        photobook_id=photobook_id,
                        user_id=request_context.user_id,
                        body=payload.body,
                        status=CommentStatus.VISIBLE,
                        notification_status=NotificationStatus.PENDING,
                    ),
                )

        return CreateCommentResponse(
            comment=PhotobookCommentsOverviewResponse.from_dao(comment)
        )

    @enforce_response_model
    async def photobook_edit_comment(
        self,
        request: Request,
        photobook_id: UUID,
        comment_id: UUID,
        payload: EditCommentRequest,
    ) -> EditCommentResponse:
        request_context = await self.get_request_context(request)

        async with self.app.new_db_session() as db_session:
            comment = await DALPhotobookComments.get_by_id(db_session, comment_id)
            if comment is None or comment.photobook_id != photobook_id:
                raise HTTPException(status_code=404, detail="Comment not found")

            if comment.user_id != request_context.user_id:
                raise HTTPException(
                    status_code=403, detail="You can only edit your own comment"
                )

            async with safe_commit(db_session):
                updated_comment = await DALPhotobookComments.update_by_id(
                    db_session,
                    comment_id,
                    DAOPhotobookCommentsUpdate(
                        body=payload.body,
                        last_updated_by=request_context.user_id,
                    ),
                )

        return EditCommentResponse(
            comment=PhotobookCommentsOverviewResponse.from_dao(updated_comment)
        )
