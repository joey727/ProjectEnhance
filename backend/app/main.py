from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from backend.app.config import settings
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()
app.add_middleware(SessionMiddleware, os.environ.get(
    "FASTAPI_SECRET_KEY", "random_secret_key"))

# Serve static files (HTML, CSS, JS, etc.)
app.mount("/landing", StaticFiles(directory="landing/public",
          html=True), name="static")

oauth = OAuth()
oauth.register(
    name='google',
    client_id=settings.client_id,
    client_secret=settings.client_secret,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile',
                   'redirect_url': 'http://127.0.0.1:8000'}
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/landing/")


@app.get("/")
async def homepage(request: Request):
    with open("landing/public/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


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
        return RedirectResponse(url="/landing/dashboard.html")


# logout route

@app.get('/logout')
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url="/landing/login.html")
