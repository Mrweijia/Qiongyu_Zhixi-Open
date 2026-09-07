"""HTTP API: upload data, validate continuity, run real T+1..T+3 forecasts.

v2 wiring (audit fixes 5.1 / 5.3 / 5.4 / 5.7):
* inference uses the TRAINING-ONLY scaler stored inside the model bundle
  (no more per-upload mean/std);
* T+1/T+2/T+3 come from ONE direct multi-step forward pass - genuinely
  different values, with real future timestamps;
* upload validation checks the 12h continuity + weather-station data contract.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
import shutil
import sys
import uuid

from flask import Blueprint, current_app, jsonify, request, send_file
import pandas as pd
from werkzeug.exceptions import RequestEntityTooLarge

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import pipeline as pl  # noqa: E402
from src.eval.risk_grading import describe, get_rule, grade, summarize  # noqa: E402
from src.inference.predictor_v2 import (  # noqa: E402
    InputValidationError,
    load_bundle,
)

predict_bp = Blueprint('predict', __name__)

UPLOAD_FOLDER = ROOT_DIR / 'outputs' / 'runtime' / 'uploads'
RESULTS_FOLDER = ROOT_DIR / 'outputs' / 'runtime' / 'results'
BUNDLE_DIR = ROOT_DIR / 'models' / 'checkpoints' / 'multistep_v2'
ALLOWED_EXTENSIONS = {'csv'}
WEB_INPUT_WEATHER_TIMEZONE = 'Asia/Shanghai'
# Upload sessions and their result CSVs must not accumulate forever: uptime x
# anonymous traffic would otherwise turn RAM and the runtime disk unbounded.
SESSION_TTL_SECONDS = 60 * 60

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

upload_status = {}
_predictor = None
_predictor_lock = Lock()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def cleanup_expired_sessions(remove_files: bool = True):
    """Drop sessions past SESSION_TTL_SECONDS, optionally their folder + result CSV."""
    now = datetime.now(timezone.utc)
    for session_id in list(upload_status):
        info = upload_status[session_id]
        expires_at = info.get('expires_at')
        if not expires_at or datetime.fromisoformat(expires_at) >= now:
            continue
        upload_status.pop(session_id, None)
        if not remove_files:
            continue
        folder = UPLOAD_FOLDER / session_id
        shutil.rmtree(folder, ignore_errors=True)
        result = RESULTS_FOLDER / f'{session_id}_predictions.csv'
        result.unlink(missing_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_predictor():
    global _predictor
    with _predictor_lock:
        if _predictor is None:
            _predictor = load_bundle(BUNDLE_DIR)
    return _predictor


def validate_inputs(pollution_path, weather_path):
    """Contract + continuity check; raises with a user-readable Chinese message."""
    try:
        tl = pl.build_timeline_arrays(
            pollution_path,
            weather_path,
            weather_timezone=WEB_INPUT_WEATHER_TIMEZONE,
        )
    except pl.DataContractError as exc:
        raise ValueError(str(exc))
    if len(tl.times) < pl.INPUT_STEPS:
        raise ValueError(f'数据不足：整点网格仅覆盖 {len(tl.times)} 小时，至少需要连续 {pl.INPUT_STEPS} 小时观测')
    try:
        pl.last_valid_window(tl)
    except pl.DataContractError as exc:
        raise ValueError(str(exc))


@predict_bp.get('/health')
def health():
    """Lightweight endpoint used by Cloudflare and deployment checks.

    FLASH-02: readiness now requires the full bundle (manifest + model.pt +
    scaler.npz) to be present and coherent via the shared model_bundle checker,
    so a missing scaler or corrupted manifest can no longer report ready (F-018).
    """
    from app.backend.model_bundle import check_bundle
    cleanup_expired_sessions(remove_files=False)
    ok, reason, manifest = check_bundle(BUNDLE_DIR)
    version = None
    if ok and manifest:
        version = manifest.get('model_sha256')
    return jsonify({
        'status': 'ok' if ok else 'degraded',
        'service': 'qiongyu-zhixi-api',
        'model_ready': ok,
        'model': 'multistep_v2 (T+1..T+3 direct multi-step)',
        'model_version': version,
        'reason': reason,
        'timestamp': utc_now(),
    }), 200 if ok else 503


@predict_bp.get('/samples/<filename>')
def sample_file(filename):
    """Expose the two public demo inputs without exposing arbitrary paths."""
    if filename not in {'pollution.csv', 'weather.csv'}:
        return jsonify({'error': '示例文件不存在'}), 404
    path = ROOT_DIR / 'data' / 'samples' / filename
    if not path.exists():
        return jsonify({'error': '示例文件不存在'}), 404
    return send_file(path, as_attachment=True, download_name=filename, mimetype='text/csv')


@predict_bp.post('/upload')
def upload_files():
    """Validate and store one pollution CSV and one weather CSV."""
    session_folder = None
    try:
        content_length = request.content_length
        max_content_length = current_app.config.get('MAX_CONTENT_LENGTH')
        if (content_length is not None and max_content_length
                and content_length > max_content_length):
            return jsonify({'error': '文件总大小超过 25 MB，请压缩或拆分后上传'}), 413

        if 'pollution_file' not in request.files or 'weather_file' not in request.files:
            return jsonify({'error': '请同时上传污染数据和天气数据 CSV 文件'}), 400

        pollution_file = request.files['pollution_file']
        weather_file = request.files['weather_file']
        if not pollution_file.filename or not weather_file.filename:
            return jsonify({'error': '请选择两个 CSV 文件'}), 400
        if not (allowed_file(pollution_file.filename) and allowed_file(weather_file.filename)):
            return jsonify({'error': '只支持 .csv 文件'}), 400

        session_id = str(uuid.uuid4())
        session_folder = UPLOAD_FOLDER / session_id
        session_folder.mkdir(parents=True)
        pollution_path = session_folder / 'pollution.csv'
        weather_path = session_folder / 'weather.csv'
        pollution_file.save(pollution_path)
        weather_file.save(weather_path)

        pollution_df = pd.read_csv(pollution_path)
        weather_df = pd.read_csv(weather_path)
        if pollution_df.empty or weather_df.empty:
            raise ValueError('文件为空')

        validate_inputs(pollution_path, weather_path)

        upload_status[session_id] = {
            'status': 'uploaded',
            'pollution_file': str(pollution_path),
            'weather_file': str(weather_path),
            'upload_time': utc_now(),
            'expires_at': (datetime.now(timezone.utc)
                           + timedelta(seconds=SESSION_TTL_SECONDS)).isoformat(),
        }
        return jsonify({
            'message': '文件校验并上传成功（已确认 12 小时连续性与站点完整性）',
            'session_id': session_id,
            'pollution_rows': len(pollution_df),
            'weather_rows': len(weather_df),
        })
    except RequestEntityTooLarge:
        if session_folder:
            shutil.rmtree(session_folder, ignore_errors=True)
        return jsonify({'error': '文件总大小超过 25 MB，请压缩或拆分后上传'}), 413
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        if session_folder:
            shutil.rmtree(session_folder, ignore_errors=True)
        return jsonify({'error': f'文件格式错误：{exc}'}), 400
    except Exception as exc:
        if session_folder:
            shutil.rmtree(session_folder, ignore_errors=True)
        return jsonify({'error': f'上传失败：{exc}'}), 500


def build_prediction_payload(predictor, result, session_id, write_result: bool = True):
    """Grade one PredictionResult into the shared /predict response dict.

    write_result=False skips persisting the per-cell CSV; the caller must then
    clear ``download_url`` because /api/download has nothing to serve.
    """
    rule = get_rule()
    rows = []
    risk_grid = {}
    focus_values = {}
    for step_idx in range(result.values.shape[0]):
        prediction_time = result.times[step_idx].isoformat()
        hour_key = f'hour_{step_idx + 1}'
        risk_grid[hour_key] = {}
        for station_index, station in enumerate(result.stations):
            for pollutant_index, pollutant in enumerate(result.pollutants):
                value = float(result.values[step_idx, station_index, pollutant_index])
                graded = grade(rule, pollutant, value)
                rows.append({
                    'station': station,
                    'pollutant': pollutant,
                    'predicted_value': value,
                    'prediction_time': prediction_time,
                    'hour_ahead': step_idx + 1,
                    'risk_key': graded['key'] if graded else '',
                    'risk_label': graded['label'] if graded else '未分级',
                    'risk_advice': graded['advice'] if graded else '',
                    'grading_rule': rule.rule_ref,
                })
                if graded:
                    risk_grid[hour_key].setdefault(station, {})[pollutant] = {
                        'key': graded['key'],
                        'label': graded['label'],
                        'advice': graded['advice'],
                        'value': graded['value'],
                    }
                    if pollutant == rule.focus_pollutant:
                        focus_values.setdefault(hour_key, []).append(graded['value'])

    result_file = RESULTS_FOLDER / f'{session_id}_predictions.csv'
    if write_result:
        pd.DataFrame(rows).to_csv(result_file, index=False)

    result_summary = {}
    for hour in range(1, result.values.shape[0] + 1):
        hour_data = [item for item in rows if item['hour_ahead'] == hour]
        result_summary[f'hour_{hour}'] = {}
        for station in result.stations:
            station_data = [item for item in hour_data if item['station'] == station]
            result_summary[f'hour_{hour}'][station] = {
                item['pollutant']: item['predicted_value'] for item in station_data
            }

    # Overall page grade per horizon, aggregated from the rule's focus pollutant.
    risk_summary = {hour_key: summarize(rule, values)
                    for hour_key, values in focus_values.items()}

    manifest = predictor.manifest
    return {
        'message': '预测完成（T+1/T+2/T+3 为模型直接多步输出，各步独立）',
        'session_id': session_id,
        'predictions': result_summary,
        'risk': risk_grid,
        'risk_summary': risk_summary,
        'download_url': f'/api/download/{session_id}',
        'total_predictions': len(rows),
        'meta': {
            'model': 'multistep_v2',
            'model_version': manifest.get('model_sha256', '')[:12],
            'pipeline_revision': manifest.get('pipeline_revision'),
            'weather_timezone': manifest.get('weather_timezone'),
            'input_window': f"至 {result.origin.isoformat()}",
            'forecast_times': [t.isoformat() for t in result.times],
            'train_range': manifest.get('train_time_range'),
            'scope': {
                'stations': manifest.get('stations', []),
                'pollutants': manifest.get('pred_names', []),
                'input_steps': manifest.get('architecture', {}).get('input_steps'),
            },
            'grading': describe(rule),
            'input_quality': result.quality or {},
            'disclaimer': '结果供辅助决策参考，不作为官方空气质量发布依据',
        },
    }


@predict_bp.post('/predict')
def predict():
    """One direct multi-step forward: genuine T+1/T+2/T+3 for 10 stations x 4 pollutants."""
    session_id = None
    try:
        cleanup_expired_sessions()
        data = request.get_json(silent=True) or {}
        session_id = data.get('session_id')
        if not session_id or session_id not in upload_status:
            return jsonify({'error': '会话已失效，请重新上传文件'}), 400

        session_info = upload_status[session_id]
        if session_info['status'] != 'uploaded':
            return jsonify({'error': '该任务未就绪或正在处理中'}), 409
        upload_status[session_id]['status'] = 'processing'

        predictor = get_predictor()
        result = predictor.predict_with_timeline(
            session_info['pollution_file'], session_info['weather_file'])
        payload = build_prediction_payload(predictor, result, session_id)
        upload_status[session_id].update({
            'status': 'completed',
            'result_file': str(RESULTS_FOLDER / f'{session_id}_predictions.csv'),
            'prediction_time': utc_now(),
            'total_predictions': payload['total_predictions'],
        })
        return jsonify(payload)
    except InputValidationError as exc:
        if session_id in upload_status:
            upload_status[session_id]['status'] = 'uploaded'
        return jsonify({'error': f'数据校验失败：{exc}'}), 400
    except (ValueError, KeyError) as exc:
        if session_id in upload_status:
            upload_status[session_id]['status'] = 'uploaded'
        return jsonify({'error': f'数据处理失败：{exc}'}), 400
    except Exception as exc:
        if session_id in upload_status:
            upload_status[session_id]['status'] = 'error'
        return jsonify({'error': f'预测失败：{exc}'}), 500


@predict_bp.get('/download/<session_id>')
def download_results(session_id):
    """Download a completed prediction as CSV."""
    cleanup_expired_sessions()
    session_info = upload_status.get(session_id)
    if not session_info:
        return jsonify({'error': '会话已失效'}), 400
    if session_info['status'] != 'completed':
        return jsonify({'error': '预测尚未完成'}), 409

    result_file = Path(session_info['result_file'])
    if not result_file.exists():
        return jsonify({'error': '结果文件不存在'}), 404
    return send_file(
        result_file,
        as_attachment=True,
        download_name=f'pollution_predictions_{session_id}.csv',
        mimetype='text/csv',
    )


@predict_bp.get('/status/<session_id>')
def get_status(session_id):
    """Return public task status without disclosing local file paths."""
    cleanup_expired_sessions()
    session_info = upload_status.get(session_id)
    if not session_info:
        return jsonify({'error': '会话已失效'}), 400
    public_status = session_info.copy()
    for key in ('pollution_file', 'weather_file', 'result_file'):
        public_status.pop(key, None)
    return jsonify(public_status)
