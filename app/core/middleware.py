from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from app.core.logging import trace_id_ctx
import uuid

class TraceIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Check for header, otherwise generate new UUID
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
        
        # Set context variable
        token = trace_id_ctx.set(trace_id)
        
        try:
            response = await call_next(request)
            # Inject trace ID into response headers
            response.headers["X-Trace-ID"] = trace_id
            return response
        finally:
            # Reset context variable (clean up)
            trace_id_ctx.reset(token)
