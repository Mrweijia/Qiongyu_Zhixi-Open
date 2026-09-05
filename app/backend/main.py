import os
import secrets
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flask import Flask, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from app.backend.predict import predict_bp
from app.backend.online import online_bp
from app.backend.observations import observations_bp

FRONTEND_DIR = ROOT_DIR / 'app' / 'frontend'

# 部署形态：web = 公网服务器；local = 软件版本地工作台（只绑 127.0.0.1）
PROFILE_WEB = 'web'
PROFILE_LOCAL = 'local'
VALID_PROFILES = (PROFILE_WEB, PROFILE_LOCAL)


def create_app(profile: str = PROFILE_WEB) -> Flask:
    """Build the Flask app for one deployment profile.

    Both profiles serve the same frontend and the same API so that one model
    bundle can power the website and the local software workbench without a
    second implementation. The local profile only tightens the network surface:
    it never trusts a shared secret key and it does not open cross-origin
    access, because a local workbench has no other origin to talk to.
    """
    if profile not in VALID_PROFILES:
        raise ValueError(f'unknown profile {profile!r}, expected one of {VALID_PROFILES}')

    app = Flask(__name__, static_folder=str(FRONTEND_DIR))
    app.config.update(
        QIONGYU_PROFILE=profile,
        SECRET_KEY=os.environ.get('FLASK_SECRET_KEY') or secrets.token_hex(32),
        MAX_CONTENT_LENGTH=25 * 1024 * 1024,
        JSON_AS_ASCII=False,
    )

    if profile == PROFILE_WEB:
        # Cloudflare Worker 使用同源转发；环境变量可为本地调试额外开放来源。
        allowed_origins = os.environ.get(
            'ALLOWED_ORIGINS',
            'https://zhixi.weijiahub.top,http://localhost:5000',
        ).split(',')
        CORS(app, resources={r'/api/*': {'origins': allowed_origins}})
    else:
        # 本地工作台是同源单页应用，不需要任何跨域放行。
        app.config['ALLOWED_ORIGINS'] = []

    app.register_blueprint(predict_bp, url_prefix='/api')
    app.register_blueprint(online_bp, url_prefix='/api')
    app.register_blueprint(observations_bp, url_prefix='/api')

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(error):
        return {'error': '请求体超过 25 MB 上限，请压缩或拆分后上传'}, 413

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        static_folder_path = app.static_folder
        if static_folder_path is None:
            return 'Static folder not configured', 404

        if path != '' and os.path.exists(os.path.join(static_folder_path, path)):
            return send_from_directory(static_folder_path, path)
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        return 'index.html not found', 404

    return app


# 向后兼容：既有代码与测试仍使用 from app.backend.main import app
app = create_app(os.environ.get('QIONGYU_PROFILE', PROFILE_WEB))


def run_local(host: str = '127.0.0.1', port: int = 0, **kwargs):
    """Start the local software workbench. Kept separate so import stays pure."""
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('本地工作台只能绑定回环地址')
    local_app = create_app(PROFILE_LOCAL)
    local_app.run(host=host, port=port, debug=False, **kwargs)


if __name__ == '__main__':
    if PROFILE_LOCAL in sys.argv:
        run_local(port=int(os.environ.get('PORT', '0') or 0))
    else:
        app.run(host='0.0.0.0', port=5000, debug=False)
