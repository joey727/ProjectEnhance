from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from backend.app.config import settings
from fastapi.staticfiles import StaticFiles
import os
import logging
import shutil
import uuid
from datetime import datetime
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.access")
logger.setLevel(logging.WARNING)


app = FastAPI()
app.add_middleware(SessionMiddleware, os.environ.get(
    "FASTAPI_SECRET_KEY", "random_secret_key"))

# Set up templates
templates = Jinja2Templates(directory="landing/public")

# Serve static files (HTML, CSS, JS, etc.)
app.mount("/landing", StaticFiles(directory="landing/public",
          html=True), name="static")


oauth = OAuth()
oauth.register(
    name='google',
    client_id=settings.client_id,
    client_secret=settings.client_secret,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile',
        'redirect_url': 'http://127.0.0.1:8000',
        'debug': False
    }
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/landing/")


@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for('auth')
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth")
async def auth(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = await oauth.google.userinfo(token=token)
    if user_info:
        request.session['user'] = dict(user_info)
        return RedirectResponse(url="/dashboard")


# logout route

@app.get('/logout')
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url="/landing/login.html?logout=true")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url="/login")
    files = request.session.get("files", [])
    credits = request.session.get("credits", 10)
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user, "files": files, "credits": credits})


@app.get("/api/credits")
async def get_credits(request: Request):
    credits = request.session.get("credits", 10)
    return {"credits": credits}

# File upload directories
UPLOAD_DIR = "landing/public/uploads"
ENHANCED_DIR = "landing/public/enhanced"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ENHANCED_DIR, exist_ok=True)


@app.post("/api/enhance")
async def enhance_image(
    request: Request,
    file: UploadFile = File(...),
    enhancement_type: str = Form("all")
):
    user = request.session.get('user')
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Save uploaded file
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # --- Simple Unblur (Deblur) using OpenCV ---
    # Read the image
    image = cv2.imdecode(np.fromfile(
        file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # Apply a sharpening kernel (simple deblurring)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    enhanced_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

    # Save enhanced image
    enhanced_filename = f"enhanced_{filename}"
    enhanced_path = os.path.join(ENHANCED_DIR, enhanced_filename)
    cv2.imwrite(enhanced_path, enhanced_image)
    # -------------------------------------------

    # Save file info to session (or DB in production)
    files = request.session.get("files", [])
    files.append({
        "filename": file.filename,
        "enhanced_url": f"/landing/enhanced/{enhanced_filename}",
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "enhancement_type": enhancement_type
    })
    request.session["files"] = files

    return {
        "enhanced_url": f"/landing/enhanced/{enhanced_filename}",
        "credits_used": 1
    }
