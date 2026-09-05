const selected={
  pollution:null,weather:null
};

let sessionId=null,predictionData=null,toastTimer=null,trainingTimer=null;

// Risk grades come from the server-side rule file: the UI never invents thresholds.
let predictionRisk=null,predictionRiskSummary=null,predictionGrading=null,predictionQuality=null;
let windowPredictionData=null,windowRisk=null,windowRiskSummary=null,windowGrading=null;

const stations=['1335A','1336A','1337A','1338A','1339A','1340A','1341A','1342A','1343A','1344A'];

const pollutants=['PM2.5','PM10','NO2','O3'];

const seriesColors={
  'PM2.5':'#78b94a',PM10:'#1ba89a',NO2:'#e49c39',O3:'#ca6371'
};

const ui=Object.fromEntries(['pollutionInput','weatherInput','pollutionDrop','weatherDrop','pollutionMeta','weatherMeta','predictBtn','resetBtn','sampleBtn','actionTitle','actionDetail','progressBar','results','resultsBody','predictionCount','pm25Average','riskLevel','riskDetail','qualityState','qualityDetail','trendChart','chartHour','downloadBtn','serviceStatus','statusDot','toast','refreshWindowBtn','observationToken','observationPayload','pushObservationsBtn','predictWindowBtn','windowStatusBadge','windowCoverage','windowRejections','windowRejectionCount','windowRejectionBody','windowPrediction','windowSourceBadge','windowPredictionCount','windowPm25Average','windowRiskLevel','windowRiskDetail','windowDataSource','windowChartHour','windowTrendChart','windowResultsBody','trainingForm','trainingFile','trainingFileHint','trainBtn','trainStatusBadge','jobEmpty','jobDetails','jobId','jobStatus','jobUpdated','trainingMetrics','metricsBody','trainingDownload','refreshModels','modelGrid','copyCode','apiCode'].map(id=>[id,document.querySelector(`#${id}`)]));

const missingUI=Object.entries(ui).filter(([,node])=>!node).map(([id])=>id);

if(missingUI.length)console.error(`Missing UI element ids: ${missingUI.join(', ')}`);



document.addEventListener('DOMContentLoaded',()=>{
  setupNavigation();

  setupDropZone('pollution');

  setupDropZone('weather');

  ui.predictBtn.addEventListener('click',runPrediction);

  ui.resetBtn.addEventListener('click',resetWorkspace);

  ui.sampleBtn.addEventListener('click',loadSamples);

  document.querySelectorAll('.horizon-tab').forEach(b=>b.addEventListener('click',()=>selectHour(Number(b.dataset.hour))));

  setupHorizonKeyboard();

  ui.refreshWindowBtn.addEventListener('click',refreshWindowCoverage);

  ui.pushObservationsBtn.addEventListener('click',pushObservations);

  ui.predictWindowBtn.addEventListener('click',predictFromWindow);

  document.querySelectorAll('.window-horizon-tab').forEach(b=>b.addEventListener('click',()=>selectWindowHour(Number(b.dataset.windowHour))));

  ui.trainingForm.addEventListener('submit',createTrainingJob);

  ui.refreshModels.addEventListener('click',loadModels);

  ui.copyCode.addEventListener('click',copyExample);

  checkService();

  refreshWindowCoverage()
}

);

function setupNavigation(){
  document.querySelectorAll('.rail-item').forEach(b=>b.addEventListener('click',()=>activateView(b.dataset.view)))
}

function activateView(name){
  document.querySelectorAll('.rail-item').forEach(i=>i.classList.toggle('active',i.dataset.view===name));

  document.querySelectorAll('[data-view-panel]').forEach(p=>{
    const active=p.dataset.viewPanel===name;

    p.hidden=!active;

    p.classList.toggle('active',active)
  }

  );

  if(name==='models')loadModels();

  if(name==='ingest')refreshWindowCoverage();

  document.querySelector('.workspace').scrollIntoView({
    behavior:'smooth',block:'start'
  }

  )
}

function setupHorizonKeyboard(){
  const tabs=[...document.querySelectorAll('.horizon-tab')];

  tabs.forEach((tab,index)=>tab.addEventListener('keydown',event=>{
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;

    event.preventDefault();

    const target=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;

    tabs[target].focus();

    tabs[target].click()
  }

  ))
}

function setupDropZone(type){
  const input=type==='pollution'?ui.pollutionInput:ui.weatherInput,drop=type==='pollution'?ui.pollutionDrop:ui.weatherDrop;

  drop.addEventListener('click',()=>input.click());

  drop.addEventListener('keydown',e=>{
    if(e.key==='Enter'||e.key===' '){
      e.preventDefault();

      input.click()
    }

  }

  );

  input.addEventListener('change',()=>setFile(type,input.files[0]));

  ['dragenter','dragover'].forEach(n=>drop.addEventListener(n,e=>{
    e.preventDefault();

    drop.classList.add('dragging')
  }

  ));

  ['dragleave','drop'].forEach(n=>drop.addEventListener(n,e=>{
    e.preventDefault();

    drop.classList.remove('dragging')
  }

  ));

  drop.addEventListener('drop',e=>setFile(type,e.dataTransfer.files[0]))
}

function setFile(type,file){
  if(!file)return;

  if(!file.name.toLowerCase().endsWith('.csv'))return showToast('请选择 CSV 格式文件。',true);

  if(file.size>25*1024*1024)return showToast('单个文件不能超过 25 MB。',true);

  selected[type]=file;

  sessionId=null;

  const drop=type==='pollution'?ui.pollutionDrop:ui.weatherDrop,meta=type==='pollution'?ui.pollutionMeta:ui.weatherMeta;

  drop.classList.add('ready');

  meta.textContent=`已选择：${file.name} · ${formatBytes(file.size)}`;

  updateReadiness()
}

function updateReadiness(){
  const ready=Boolean(selected.pollution&&selected.weather),count=Number(Boolean(selected.pollution))+Number(Boolean(selected.weather));

  ui.predictBtn.disabled=!ready;

  ui.actionTitle.textContent=ready?'数据已就绪':count?'还需要 1 份数据':'等待两份数据';

  ui.actionDetail.textContent=ready?'运行前将校验时间连续性与站点数据契约。':count?'请继续选择另一份 CSV 文件。':'至少需要连续 12 小时观测，可先载入示例验证流程。'
}

async function loadSamples(){
  setBusy(true,'正在读取示例数据…');

  try{
    const responses=await Promise.all(['/api/samples/pollution.csv','/api/samples/weather.csv'].map(url=>fetch(url)));

    if(responses.some(r=>!r.ok))throw new Error('示例文件不可用');

    const blobs=await Promise.all(responses.map(r=>r.blob()));

    setFile('pollution',new File([blobs[0]],'pollution.csv',{
      type:'text/csv'
    }

    ));

    setFile('weather',new File([blobs[1]],'weather.csv',{
      type:'text/csv'
    }

    ));

    showToast('示例数据已载入，可以运行预测。')
  }

  catch(error){
    showToast(`载入失败：${error.message}`,true)
  }

  finally{
    setBusy(false);

    updateReadiness()
  }

}

async function runPrediction(){
  if(!selected.pollution||!selected.weather)return;

  ui.results.hidden=true;

  ui.predictBtn.disabled=true;

  setBusy(true,'正在上传并校验…');

  setProgress(18);

  setQuality('校验中','正在检查 CSV 格式与连续性','neutral');

  try{
    const form=new FormData();

    form.append('pollution_file',selected.pollution);

    form.append('weather_file',selected.weather);

    const upload=await apiRequest('/api/upload',{
      method:'POST',body:form
    }

    );

    sessionId=upload.session_id;

    setProgress(55);

    setQuality('通过',`已校验 ${upload.pollution_rows} 行污染 / ${upload.weather_rows} 行气象数据`,'low');

    ui.actionTitle.textContent='模型正在推理';

    ui.actionDetail.textContent='正在生成 T+1、T+2、T+3 的直接多步输出。';

    const result=await apiRequest('/api/predict',{
      method:'POST',headers:{
        'Content-Type':'application/json'
      }

      ,body:JSON.stringify({
        session_id:sessionId
      }

      )
    }

    );

    predictionData=result.predictions;

    predictionRisk=result.risk||null;

    predictionRiskSummary=result.risk_summary||null;

    predictionGrading=result.meta?.grading||null;

    predictionQuality=result.meta?.input_quality||null;

    if(Number.isFinite(predictionQuality?.input_missing_frac)){
      setQuality('通过',`已校验 ${predictionQuality.window_hours}/${predictionQuality.required_hours} 小时输入窗口，实测缺失 ${(predictionQuality.input_missing_frac*100).toFixed(2)}%`,'low');
    }

    ui.predictionCount.textContent=`${result.total_predictions||'—'} 个`;

    ui.downloadBtn.href=result.download_url||'#';

    ui.actionTitle.textContent='预测已完成';

    ui.actionDetail.textContent=result.meta?.disclaimer||'可切换未来 1—3 小时结果，或下载完整 CSV。';

    setProgress(100);

    ui.results.hidden=false;

    selectHour(1);

    ui.results.scrollIntoView({
      behavior:'smooth',block:'start'
    }

    );

    showToast('预测完成，结果已生成。')
  }

  catch(error){
    sessionId=null;

    setProgress(0);

    setQuality('未通过',error.message,'high');

    ui.actionTitle.textContent='本次运行未完成';

    ui.actionDetail.textContent=error.message;

    showToast(error.message,true)
  }

  finally{
    setBusy(false);

    ui.predictBtn.disabled=!(selected.pollution&&selected.weather)
  }

}

const rejectionLabels={
  record_not_object:'记录不是对象',
  invalid_time:'时间必须是整点 ISO 8601',
  future_time:'时间不能晚于当前',
  missing_station:'缺少站点编号',
  unknown_station:'站点不在当前模型契约中',
  measurements_not_object:'污染物或气象字段不是对象',
  unknown_field:'包含模型契约外的字段',
  empty_record:'记录中没有可用观测值'
};

function rejectionText(reason){
  if(reason?.startsWith('invalid_value:'))return `数值超出允许范围：${reason.slice('invalid_value:'.length)}`;
  return rejectionLabels[reason]||reason||'未知原因'
}

function applyWindowCoverage(coverage){
  const c=coverage||{};
  const dd=[...ui.windowCoverage.querySelectorAll('dd')];
  const stored=Number(c.hours_stored||0);
  const complete=Number(c.complete_hours||0);
  const consecutive=Number(c.consecutive_complete_hours||0);
  const needed=Number(c.hours_needed||0);
  const latestAge=c.latest_age_minutes;
  const noData=stored===0;
  if(dd.length>=1)dd[0].textContent=`${stored} 小时`;
  if(dd.length>=2)dd[1].textContent=`${complete} / ${needed || 0} 小时`;
  if(dd.length>=3)dd[2].textContent=`${consecutive} / ${needed || 0} 小时`;
  if(dd.length>=4)dd[3].textContent=latestAge===null||latestAge===undefined?'暂无完整窗口':`${Math.max(0,latestAge)} 分钟前`;
  if(dd.length>=5)dd[4].textContent=c.ready_for_prediction?'是':'否';
  ui.windowStatusBadge.textContent=c.ready_for_prediction?'可预测':noData?'等待数据':'覆盖不足';
  ui.windowStatusBadge.className=`badge ${c.ready_for_prediction?'completed':noData?'muted':'pending'}`
}

function renderRejections(rejected){
  const rows=Array.isArray(rejected)?rejected:[];
  ui.windowRejectionCount.textContent=`${rows.length} 条`;
  if(!rows.length){
    ui.windowRejections.hidden=true;
    ui.windowRejectionBody.replaceChildren();
    return
  }

  ui.windowRejections.hidden=false;
  ui.windowRejectionBody.replaceChildren(...rows.map(item=>{
    const row=document.createElement('tr');
    [item.index,item.station||'—',rejectionText(item.reason)].forEach(value=>{
      const cell=document.createElement('td');
      cell.textContent=String(value);
      row.appendChild(cell)
    });
    return row
  }))
}

async function refreshWindowCoverage(){
  try{
    const payload=await apiRequest('/api/v1/observations/window');
    applyWindowCoverage(payload.coverage);
    renderRejections([]);
    return payload.coverage
  }
  catch(error){
    ui.windowStatusBadge.textContent='窗口接口不可用';
    ui.windowStatusBadge.className='badge error';
    return null
  }
}

async function pushObservations(){
  const token=ui.observationToken.value.trim();
  if(!token)return showToast('请输入 QIONGYU_API_TOKEN 后推送。',true);
  const raw=ui.observationPayload.value.trim();
  if(!raw)return showToast('请先粘贴观测 JSON。',true);
  let records;
  try{
    const parsed=JSON.parse(raw);
    records=Array.isArray(parsed)?parsed:[parsed]
  }
  catch{
    return showToast('观测 JSON 格式错误，请检查逗号和引号。',true)
  }

  ui.pushObservationsBtn.disabled=true;
  ui.pushObservationsBtn.firstElementChild.textContent='正在推送…';
  try{
    const result=await apiRequest('/api/v1/observations',{
      method:'POST',
      headers:{
        'Content-Type':'application/json',
        Authorization:`Bearer ${token}`
      },
      body:JSON.stringify({observations:records})
    });
    applyWindowCoverage(result.window);
    renderRejections(result.rejected);
    if(result.accepted){
      const extra=result.rejected?.length?`，拒绝 ${result.rejected.length} 条`:'';
      showToast(`已接收 ${result.accepted} 条观测${extra}。`);
    }else{
      showToast('本次没有可接受的观测记录。',true)
    }
    await refreshWindowCoverage()
  }
  catch(error){
    if(error.payload?.window)applyWindowCoverage(error.payload.window);
    showToast(error.message,true)
  }
  finally{
    ui.pushObservationsBtn.disabled=false;
    ui.pushObservationsBtn.firstElementChild.textContent='推送观测'
  }
}

async function predictFromWindow(){
  const token=ui.observationToken.value.trim();
  if(!token)return showToast('请输入 QIONGYU_API_TOKEN 后执行窗口预测。',true);
  const original=ui.predictWindowBtn.textContent;
  ui.predictWindowBtn.disabled=true;
  ui.predictWindowBtn.textContent='正在预测…';
  try{
    const result=await apiRequest('/api/v1/observations/predict',{
      method:'POST',
      headers:{
        Authorization:`Bearer ${token}`
      }
    });
    windowPredictionData=result.predictions||{};
    windowRisk=result.risk||null;
    windowRiskSummary=result.risk_summary||null;
    windowGrading=result.meta?.grading||null;
    ui.windowPredictionCount.textContent=`${result.total_predictions||0} 个`;
    ui.windowDataSource.textContent=result.source==='observation_window'?'滚动观测窗口':'窗口预测';
    ui.windowSourceBadge.textContent='当前观察窗口';
    ui.windowPrediction.hidden=false;
    selectWindowHour(1);
    ui.windowPrediction.scrollIntoView({
      behavior:'smooth',block:'start'
    });
    showToast('窗口预测完成，结果已展示。')
  }
  catch(error){
    if(error.status===409&&error.payload?.coverage)applyWindowCoverage(error.payload.coverage);
    showToast(error.message,true)
  }
  finally{
    ui.predictWindowBtn.disabled=false;
    ui.predictWindowBtn.textContent=original
  }
}

async function apiRequest(url,options={
}

){
  let response;

  try{
    response=await fetch(url,options)
  }

  catch{
    throw new Error('无法连接预测服务，请稍后重试。')
  }

  const type=response.headers.get('content-type')||'',payload=type.includes('application/json')?await response.json():{
    error:await response.text()
  };

  if(!response.ok){
    const error=new Error(payload.error||`服务返回错误 ${response.status}`);

    error.status=response.status;

    error.payload=payload;

    throw error
  }

  return payload
}

function selectHour(hour){
  document.querySelectorAll('.horizon-tab').forEach(b=>{
    const active=Number(b.dataset.hour)===hour;

    b.classList.toggle('active',active);

    b.setAttribute('aria-selected',String(active))
  }

  );

  const values=predictionData?.[`hour_${hour}`];

  if(!values)return;

  ui.chartHour.textContent=`T+${hour}`;

  renderTable(values,hour);

  renderSummary(values,hour);

  renderTrendChart(hour)
}

function selectWindowHour(hour){
  document.querySelectorAll('.window-horizon-tab').forEach(b=>{
    const active=Number(b.dataset.windowHour)===hour;

    b.classList.toggle('active',active);

    b.setAttribute('aria-selected',String(active))
  }

  );

  const values=windowPredictionData?.[`hour_${hour}`];

  if(!values)return;

  ui.windowChartHour.textContent=`T+${hour}`;

  renderWindowTable(values,hour);

  renderWindowSummary(values,hour);

  renderWindowTrendChart(hour)
}

function renderTable(values,hour=1){
  ui.resultsBody.replaceChildren(...buildPredictionRows(values,hour,predictionRisk,predictionGrading))
}

function renderWindowTable(values,hour=1){
  ui.windowResultsBody.replaceChildren(...buildPredictionRows(values,hour,windowRisk,windowGrading))
}

function buildPredictionRows(values,hour,risk,grading){
  const hourRisk=risk?.[`hour_${hour}`]||{};
  return stations.map(station=>{
    const row=document.createElement('tr'),data=values[station]||{},stationRisk=hourRisk[station]||{};

    [[station,null],...pollutants.map(name=>[formatConcentration(data[name]),stationRisk[name]])].forEach(([value,risk])=>{
      const cell=document.createElement('td');

      cell.textContent=value;

      if(risk){
        const suffix=grading?.rule?`（规则 ${grading.rule}）`:''; 
        cell.className=`risk-cell risk-${risk.key}`;
        cell.title=`${risk.label}：${risk.advice}${suffix}`;
      }

      row.appendChild(cell)
    });

    return row
  })
}

function renderSummary(values,hour=1){
  renderPm25Average(ui.pm25Average,values);

  renderRiskBox(ui.riskLevel,ui.riskDetail,hour,predictionRiskSummary,predictionGrading)
}

function renderWindowSummary(values,hour=1){
  renderPm25Average(ui.windowPm25Average,values);

  renderRiskBox(ui.windowRiskLevel,ui.windowRiskDetail,hour,windowRiskSummary,windowGrading)
}

function renderPm25Average(element,values){
  const pm25=stations.map(s=>Number(values[s]?.['PM2.5'])).filter(Number.isFinite),average=pm25.length?pm25.reduce((a,b)=>a+b,0)/pm25.length:NaN;

  element.textContent=Number.isFinite(average)?`${average.toFixed(1)} μg/m³`:'—'
}

function renderRiskBox(level,detail,hour,riskSummary,grading){
  const server=riskSummary?.[`hour_${hour}`];
  if(server?.status==='ok'){
    const unit=grading?.unit||'ug_m3';
    const how=server.aggregation==='max'?'最差站点':'全站均值';
    level.textContent=server.label;
    level.className=`risk-${server.key}`;
    detail.textContent=`${how} ${server.aggregate_value} ${unit}｜${server.advice}｜规则 ${server.rule}｜${grading?.disclaimer||'内部辅助阈值，非官方等级'}`;
  }else if(server){
    level.textContent=server.label||'数据不足';
    level.className='risk-neutral';
    detail.textContent=server.advice||'焦点污染物没有可用预测值，暂不分级';
  }else{
    // A backend without grading must not be papered over with invented thresholds.
    level.textContent='未提供分级';
    level.className='risk-neutral';
    detail.textContent='后端未返回分级规则，页面不自造阈值';
  }
}

function renderTrendChart(currentHour){
  ui.trendChart.innerHTML=buildTrendMarkup(predictionData,currentHour)
}

function renderWindowTrendChart(currentHour){
  ui.windowTrendChart.innerHTML=buildTrendMarkup(windowPredictionData,currentHour)
}

function buildTrendMarkup(data,currentHour){
  const width=620,height=228,pad={
    top:18,right:20,bottom:30,left:42
  }

  ,hours=[1,2,3],avgs=Object.fromEntries(pollutants.map(p=>[p,hours.map(h=>{
    const v=stations.map(s=>Number(data?.[`hour_${h}`]?.[s]?.[p])).filter(Number.isFinite);

    return v.length?v.reduce((a,b)=>a+b,0)/v.length:0
  }

  )])),max=Math.max(1,...Object.values(avgs).flat())*1.12,x=h=>pad.left+(h-1)*(width-pad.left-pad.right)/2,y=v=>height-pad.bottom-v/max*(height-pad.top-pad.bottom);

  let content=[0,.25,.5,.75,1].map(part=>`<line class="chart-grid" x1="${pad.left}" x2="${width-pad.right}" y1="${y(max*part)}" y2="${y(max*part)}"/><text class="chart-label" x="2" y="${y(max*part)+4}">${(max*part).toFixed(0)}</text>`).join('');

  content+=hours.map(h=>`<text class="chart-label" text-anchor="middle" x="${x(h)}" y="${height-8}">T+${h}</text>`).join('');

  Object.entries(avgs).forEach(([name,values])=>{
    content+=`<polyline class="trend-line" stroke="${seriesColors[name]}" points="${values.map((v,i)=>`${
      x(i+1)
    }

    ,${
      y(v)
    }

    `).join(' ')}"/>`;

    values.forEach((v,i)=>content+=`<circle class="trend-dot" fill="${seriesColors[name]}" cx="${x(i+1)}" cy="${y(v)}" r="${i+1===currentHour?5:3.5}"/>`)
  }

  );

  return content
}

function setQuality(state,detail,kind){
  ui.qualityState.textContent=state;

  ui.qualityState.className=`risk-${kind}`;

  ui.qualityDetail.textContent=detail
}

function resetWorkspace(){
  selected.pollution=null;

  selected.weather=null;

  sessionId=null;

  predictionData=null;

  predictionRisk=null;

  predictionRiskSummary=null;

  predictionGrading=null;

  predictionQuality=null;

  ui.pollutionInput.value='';

  ui.weatherInput.value='';

  ui.pollutionDrop.classList.remove('ready');

  ui.weatherDrop.classList.remove('ready');

  ui.pollutionMeta.textContent='选择 pollution.csv 或将文件拖到这里';

  ui.weatherMeta.textContent='选择 weather.csv 或将文件拖到这里';

  ui.results.hidden=true;

  setProgress(0);

  setQuality('待校验','上传后检查格式与连续性','neutral');

  updateReadiness()
}

async function checkService(){
  try{
    const health=await apiRequest('/api/health');

    if(!health.model_ready)throw new Error();

    ui.statusDot.className='status-dot online';

    ui.serviceStatus.textContent='预测服务在线 · 模型已就绪'
  }

  catch{
    ui.statusDot.className='status-dot offline';

    ui.serviceStatus.textContent='预测服务暂时不可用'
  }

}

async function createTrainingJob(event){
  event.preventDefault();

  const file=ui.trainingFile.files[0],horizons=[...document.querySelectorAll('input[name="horizon"]:checked')].map(i=>i.value);

  if(!file)return showToast('请选择训练数据 CSV。',true);

  if(!horizons.length)return showToast('至少选择一个预测时段。',true);

  ui.trainingMetrics.hidden=true;

  ui.trainingDownload.hidden=true;

  ui.trainBtn.disabled=true;

  ui.trainBtn.textContent='正在创建任务…';

  const form=new FormData();

  form.append('file',file);

  form.append('target',document.querySelector('#targetSelect').value);

  form.append('horizons',horizons.join(','));

  try{
    const job=await apiRequest('/api/train',{
      method:'POST',body:form
    }

    ),jobId=job.job_id;

    if(!jobId)throw new Error('训练服务未返回任务 ID。');

    showTrainingJob(jobId,job.status||'queued',job);

    showToast('训练任务已创建，正在轮询状态。');

    pollTrainingJob(jobId)
  }

  catch(error){
    const unavailable=[404,405,501].includes(error.status);

    ui.trainingFileHint.textContent=unavailable?'当前部署尚未提供训练接口，可继续使用在线预测。':error.message;

    ui.trainStatusBadge.textContent=unavailable?'接口未部署':'创建失败';

    ui.trainStatusBadge.className='badge error';

    showToast(unavailable?'训练接口暂不可用，已保留在线预测入口。':error.message,true)
  }

  finally{
    ui.trainBtn.disabled=false;

    ui.trainBtn.innerHTML='创建 Ridge 模型 <span aria-hidden="true">→</span>'
  }

}

function showTrainingJob(id,status,data={
}

){
  ui.jobEmpty.hidden=true;

  ui.jobDetails.hidden=false;

  ui.jobId.textContent=id;

  ui.jobStatus.textContent=status;

  ui.jobUpdated.textContent=data.completed_at||data.finished_at||data.started_at||data.created_at||data.updated_at||data.updated||'等待服务更新';

  if(data.metrics?.validation)renderValidationMetrics(data.metrics.validation);

  setJobBadge(status)
}

function setJobBadge(status){
  const n=String(status).toLowerCase(),type=/(complete|success|done)/.test(n)?'completed':/(fail|error|cancel)/.test(n)?'error':/(run|process)/.test(n)?'running':'pending';

  ui.trainStatusBadge.textContent=status;

  ui.trainStatusBadge.className=`badge ${type}`
}

function renderValidationMetrics(validation){
  const rows=Object.entries(validation||{
  }
  );

  if(!rows.length)return;

  ui.trainingMetrics.hidden=false;

  ui.metricsBody.replaceChildren(...rows.sort(([left],[right])=>Number(left)-Number(right)).map(([horizon,entry])=>{
    const ridge=entry.model||{
    }
    ,persistence=entry.persistence||{
    }
    ,improvement=Number(entry.rmse_improvement),improved=Number.isFinite(improvement)?improvement>0:null,row=document.createElement('tr'),values=[`T+${horizon}`,formatMetric(ridge.mae),formatMetric(ridge.rmse),formatMetric(ridge.r2),formatMetric(persistence.mae),formatMetric(persistence.rmse),formatMetric(persistence.r2),improved===null?'未提供':improved?'是':'否'];

    values.forEach((value,index)=>{
      const cell=document.createElement('td');

      cell.textContent=value;

      if(index===7)cell.className=improved===null?'metric-unknown':improved?'metric-improved':'metric-not-improved';

      row.appendChild(cell)
    }
    );

    return row
  }
  ))
}

function formatConcentration(value){
  const n=Number(value);
  return Number.isFinite(n)?n.toFixed(1):'—'
}

function formatMetric(value){
  return Number.isFinite(Number(value))?Number(value).toFixed(3):'—'
}

function pollTrainingJob(id){
  clearTimeout(trainingTimer);

  const poll=async()=>{
    try{
      const job=await apiRequest(`/api/train/${encodeURIComponent(id)}`),status=job.status||'processing';

      showTrainingJob(id,status,job);

      if(/(complete|success|done)/i.test(status)){
        ui.trainingDownload.href=`/api/train/${encodeURIComponent(id)}/download`;

        ui.trainingDownload.hidden=false;

        showToast('训练任务已完成，可下载模型产物。');

        return
      }

      if(/(fail|error|cancel)/i.test(status)){
        showToast(job.error||'训练任务未完成。',true);

        return
      }

      trainingTimer=setTimeout(poll,3000)
    }

    catch(error){
      ui.jobUpdated.textContent=`状态读取失败：${error.message}`;

      setJobBadge('状态不可用')
    }

  };

  poll()
}

async function loadModels(){
  ui.modelGrid.innerHTML='<article class="data-card model-placeholder"><span>◇</span><h2>正在读取模型目录</h2><p>仅展示服务端明确返回的信息。</p></article>';

  try{
    const payload=await apiRequest('/api/models'),models=Array.isArray(payload)?payload:(payload.models||payload.items||[]);

    if(!models.length){
      ui.modelGrid.innerHTML='<article class="data-card model-placeholder"><span>◇</span><h2>模型目录为空</h2><p>服务已响应，但未返回可展示模型。</p></article>';

      return
    }

    ui.modelGrid.replaceChildren(...models.map(createModelCard))
  }

  catch(error){
    const unavailable=[404,405,501].includes(error.status);

    ui.modelGrid.innerHTML=`<article class="data-card model-placeholder"><span>◇</span><h2>${unavailable?'模型中心接口未部署':'模型目录暂不可用'}</h2><p>${unavailable?'不会显示未经服务端确认的模型信息。':escapeHtml(error.message)}</p></article>`
  }

}

function createModelCard(model){
  const card=document.createElement('article'),name=model.name||model.id||'未命名模型',kind=[model.kind,model.model_type].filter(Boolean).join(' / '),status=[model.status,kind].filter(Boolean).join(' / ')||'未提供',horizons=Array.isArray(model.horizons)?model.horizons.map(h=>`T+${h}`).join('、'):model.horizons||'未提供',stations=Array.isArray(model.stations)?`${model.stations.length} 个站点`:model.stations||'未提供',outputs=Array.isArray(model.outputs)?model.outputs.join('、'):model.outputs||'未提供',trainRange=Array.isArray(model.train_range)?model.train_range.join(' 至 '):model.train_range||'未提供',fields=[['模型 ID',model.id||'未提供'],['状态',status],['版本',model.version||model.model_version||'未提供'],['预测时段',horizons],['输入窗口',model.input_steps?`${model.input_steps} 小时`:model.inputs||'未提供'],['站点数',stations],['输出指标',outputs],['训练范围',trainRange],['模型指标',compactMetricValue(model.evaluation_metrics)],['创建时间',model.created_utc||'未提供']].filter(([,value])=>value!==''&&value!==null&&value!==undefined);

  card.className='data-card model-card';

  card.innerHTML=`<span class="badge muted">服务端记录</span><h2>${escapeHtml(String(name))}</h2><p>${escapeHtml(String(model.description||`${name} 已由服务端确认，仅展示清单返回的字段。`))}</p><dl>${fields.map(([k,v])=>`<div><dt>${
    k
  }

  </dt><dd>${
    escapeHtml(String(v))
  }

  </dd></div>`).join('')}</dl>`;

  return card
}

function compactMetricValue(value){
  if(value===null||value===undefined||value==='')return '未提供';
  if(typeof value==='number')return Number.isFinite(value)?value.toFixed(4):String(value);
  if(Array.isArray(value))return value.map(compactMetricValue).filter(Boolean).join('、');
  if(typeof value==='string')return value;
  if(typeof value==='object'){
    const parts=Object.entries(value).map(([key,nested])=>`${key}:${compactMetricValue(nested)}`).filter(part=>!part.endsWith('未提供'));
    return parts.length?parts.join('｜'):'未提供'
  }
  return String(value)
}

async function copyExample(){
  try{
    await navigator.clipboard.writeText(ui.apiCode.innerText);

    ui.copyCode.textContent='已复制';

    setTimeout(()=>ui.copyCode.textContent='复制',1400)
  }

  catch{
    showToast('浏览器未允许剪贴板访问，请手动复制示例。',true)
  }

}

function setBusy(busy,label){
  if(busy){
    ui.predictBtn.querySelector('.button-label').textContent=label||'处理中…';

    ui.progressBar.classList.add('indeterminate')
  }

  else{
    ui.predictBtn.querySelector('.button-label').textContent='运行模型预测';

    ui.progressBar.classList.remove('indeterminate')
  }

}

function setProgress(value){
  ui.progressBar.classList.remove('indeterminate');

  ui.progressBar.style.width=`${value}%`
}

function showToast(message,error=false){
  clearTimeout(toastTimer);

  ui.toast.textContent=message;

  ui.toast.className=`toast show${error?' error':''}`;

  toastTimer=setTimeout(()=>ui.toast.className='toast',3600)
}

function formatValue(value){
  return Number.isFinite(Number(value))?Number(value).toFixed(2):'—'
}

function formatBytes(bytes){
  return bytes<1024?`${bytes} B`:bytes<1024**2?`${(bytes/1024).toFixed(1)} KB`:`${(bytes/1024**2).toFixed(1)} MB`
}

function escapeHtml(value){
  const node=document.createElement('div');

  node.textContent=value;

  return node.innerHTML
}
