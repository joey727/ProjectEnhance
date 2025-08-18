from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from backend.app.fpn_mobilenet import deblur_image
from fastapi.staticfiles import StaticFiles
import os
import logging
import shutil
import uuid
from datetime import datetime
from dotenv import load_dotenv
from backend.app.model_loader import load_model
from backend.app.config import settings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.access")
logger.setLevel(logging.WARNING)


app = FastAPI()

load_dotenv()

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET_KEY",
                              "supersecretkey"),
    same_site="lax",
    https_only=False
)

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

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
    if not token:
        return JSONResponse({"error": "Authentication failed"}, status_code=401)
    user_info = await oauth.google.userinfo(token=token)
    if user_info:
        request.session['user'] = dict(user_info)
        return RedirectResponse(url="/dashboard")

# @app.get("/auth")
# async def auth(request: Request):
#     token = await oauth.google.authorize_access_token(request)
#     user_info = await oauth.google.parse_id_token(request, token)
#     if user_info:
#         request.session['user'] = dict(user_info)
#         return RedirectResponse(url="/dashboard")


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

# deblurgan model

deblurgan_model = None
try:
    model = load_model(
        "backend/app/trained_models/mobilenetv2_rgb_epoch150_bs256.pth")
    print("weights successfully loaded")
except Exception as e:
    print("Failed to load PyTorch DeblurGAN model:", e)
    # Fallback to TensorFlow model
    try:
        from backend.app.model.deblurgan_predict import DeblurGANPredictor
        deblurgan_model = DeblurGANPredictor('backend/app/DeblurGAN_mode')
        print("TensorFlow DeblurGAN model loaded as fallback.")
    except Exception as e2:
        print("Failed to load TensorFlow DeblurGAN model:", e2)
        # Final fallback to enhanced OpenCV-based enhancer
        try:
            from backend.app.simple_enhancer_fixed import SimpleImagePredictor
            deblurgan_model = SimpleImagePredictor()
            print("Enhanced OpenCV-based image enhancer loaded as final fallback.")
        except Exception as e3:
            print("Failed to load simple enhancer:", e3)


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

    enhanced_filename = f"enhanced_{filename}"
    enhanced_path = os.path.join(ENHANCED_DIR, enhanced_filename)

    if model:
        deblur_image(model, file_path, enhanced_path)

    # try:
    #     if deblurgan_model:
    #         pil_image = Image.open(file_path).convert("RGB")

    #         if hasattr(deblurgan_model, 'predict') and callable(deblurgan_model.predict):
    #             # Count external arguments (excluding `self`)
    #             arg_count = deblurgan_model.predict.__code__.co_argcount - 1

    #             if arg_count == 2:
    #                 output_image = deblurgan_model.predict(
    #                     pil_image, enhancement_type)
    #             elif arg_count == 1:
    #                 output_image = deblurgan_model.predict(pil_image)
    #             else:
    #                 raise TypeError("Unsupported predict() signature.")
    #         else:
    #             output_image = deblurgan_model.predict(pil_image)

    #         output_image.save(enhanced_path)

    #     else:
    #         # Fallback: Enhanced OpenCV processing
    #         pil_image = Image.open(file_path).convert("RGB")
    #         enhancer = SimpleImageEnhancer()
    #         output_image = enhancer.enhance_image(pil_image, enhancement_type)
    #         output_image.save(enhanced_path)
    # except Exception as e:
    #     print("Enhancement error:", e)
    #     return JSONResponse({"error": "Enhancement failed"}, status_code=500)

    # Save file info to session
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
