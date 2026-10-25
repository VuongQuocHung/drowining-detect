import { useEffect, useRef, useState } from 'react';
import { API_URL, PersonDetection, SessionEvent, SessionInfo, sessionRequest, stopSession } from '../services/sessions';

const terminal = (status: string) => ['completed', 'stopped', 'failed'].includes(status);
const levelNames: Record<string, string> = {
  normal: 'Bình thường', suspected: 'Đang xác nhận', struggling_alert: 'Cảnh báo vùng vẫy',
  drowning_alert: 'Cảnh báo đuối nước', lost: 'Mất dấu — chưa rõ trạng thái',
};
const classNames: Record<string, string> = {
  swimming: 'Bơi bình thường', struggling: 'Vùng vẫy', drowning: 'Đuối nước', unknown: 'Không rõ',
};

export default function DrowningDetection() {
  const [model, setModel] = useState('underwater');
  const [mode, setMode] = useState<'video' | 'webcam' | 'image'>('video');
  const [file, setFile] = useState<File | null>(null);
  const [enhance, setEnhance] = useState(false);
  const [confidence, setConfidence] = useState(0.5);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [event, setEvent] = useState<SessionEvent | null>(null);
  const [image, setImage] = useState<string | null>(null);
  const [imagePeople, setImagePeople] = useState<PersonDetection[]>([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [connected, setConnected] = useState(false);
  const [sound, setSound] = useState(false);
  const soundRef = useRef(false);
  const audioRef = useRef<AudioContext | null>(null);
  const alarmSequence = useRef(0);
  const active = session !== null && !terminal(session.status);
  const sessionId = session?.session_id;

  useEffect(() => {
    let cancelled = false;
    void sessionRequest<SessionInfo | null>('/sessions/active').then(info => {
      if (!cancelled && info) {
        setSession(info); setModel(info.model_type); setMode(info.source);
        setEnhance(info.enhancement); setConfidence(info.confidence_threshold);
      }
    }).catch(err => { if (!cancelled) setError(String(err)); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    const source = new EventSource(`${API_URL}/sessions/${sessionId}/events`);
    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);
    source.onmessage = message => {
      const next = JSON.parse(message.data) as SessionEvent;
      setEvent(previous => ({ ...previous, ...next, metrics: next.metrics || previous?.metrics }));
      setSession(previous => previous ? { ...previous, status: next.status,
        download_available: next.status === 'completed' && previous.source === 'video',
        processing_complete: next.status === 'completed', error: next.error } : previous);
      if (next.error) setError(next.error);
      if (next.sequence > alarmSequence.current && soundRef.current && next.detections.some(p => p.alarm)) {
        const context = audioRef.current;
        if (context?.state === 'running') {
          const oscillator = context.createOscillator(); const gain = context.createGain();
          oscillator.frequency.value = 880; gain.gain.value = 0.15;
          oscillator.connect(gain); gain.connect(context.destination);
          oscillator.start(); oscillator.stop(context.currentTime + 0.6);
        }
      }
      alarmSequence.current = Math.max(alarmSequence.current, next.sequence);
      if (next.terminal) { source.close(); setConnected(false); }
    };
    return () => { source.close(); setConnected(false); };
  }, [sessionId]);

  useEffect(() => () => { void audioRef.current?.close(); }, []);
  const enableSound = async () => {
    try {
      if (!audioRef.current) audioRef.current = new AudioContext();
      await audioRef.current.resume(); soundRef.current = !soundRef.current; setSound(soundRef.current);
    } catch (err) { setError(String(err)); }
  };
  const start = async () => {
    setBusy(true); setError(''); setImage(null); setEvent(null); setImagePeople([]); alarmSequence.current = 0;
    try {
      const form = new FormData(); form.set('model_type', model);
      form.set('enhancement', String(enhance && model === 'underwater')); form.set('confidence_threshold', String(confidence));
      if (mode !== 'webcam') {
        if (!file) throw new Error('Chọn tệp trước khi chạy.');
        form.set('file', file);
      }
      if (mode === 'image') {
        form.set('source', 'image');
        const result = await sessionRequest<{ image_base64: string; detections: PersonDetection[] }>('/predict', form);
        setImage(`data:image/jpeg;base64,${result.image_base64}`); setImagePeople(result.detections); setSession(null);
      } else {
        const result = await sessionRequest<{ session_id: string }>(mode === 'webcam' ? '/webcam/start' : '/realtime_video_process', form);
        setSession(await sessionRequest<SessionInfo>(`/video_session_status/${result.session_id}`));
      }
    } catch (err) { setError(String(err)); }
    finally { setBusy(false); }
  };
  const stop = async () => {
    if (!session) return;
    setBusy(true);
    try { setSession(await stopSession(session.session_id)); }
    catch (err) { setError(String(err)); }
    finally { setBusy(false); }
  };
  const people = image ? imagePeople : event?.detections || [];
  const metrics = event?.metrics;
  const inputStyle = 'border rounded px-3 py-2 bg-white disabled:opacity-50';
  return <div className="max-w-7xl mx-auto p-6 space-y-5">
    <div><h2 className="text-2xl font-bold text-slate-900">Giám sát và nhận diện AI</h2>
      <p className="text-slate-600">Một nguồn mỗi lần · Theo dõi từng người · Cảnh báo từ kết quả mô hình</p></div>
    <div className="bg-white rounded-xl p-5 shadow space-y-4">
      <div className="flex flex-wrap gap-3">
        <select aria-label="Nguồn" className={inputStyle} value={mode} disabled={active || busy} onChange={e => { setMode(e.target.value as typeof mode); setFile(null); }}>
          <option value="video">Video</option><option value="webcam">Webcam trên máy backend</option><option value="image">Ảnh tĩnh</option>
        </select>
        <select aria-label="Mô hình" className={inputStyle} value={model} disabled={active || busy} onChange={e => setModel(e.target.value)}>
          <option value="underwater">Dưới nước</option><option value="above-water">Trên mặt nước</option>
        </select>
        <label className="flex items-center gap-2">Confidence<input aria-label="Confidence" type="number" min="0" max="1" step="0.05" className={`${inputStyle} w-24`} value={confidence} disabled={active || busy} onChange={e => setConfidence(Number(e.target.value))} /></label>
        {model === 'underwater' && <label className="flex items-center gap-2"><input type="checkbox" checked={enhance} disabled={active || busy} onChange={e => setEnhance(e.target.checked)} />Tăng cường ảnh GAN</label>}
      </div>
      {mode !== 'webcam' && <input key={mode} aria-label="Tệp đầu vào" type="file" accept={mode === 'image' ? 'image/*' : 'video/*'} disabled={active || busy} onChange={e => setFile(e.target.files?.[0] || null)} />}
      <div className="flex flex-wrap gap-3">
        <button className="bg-teal-700 text-white rounded px-4 py-2 disabled:opacity-50" disabled={active || busy || (mode !== 'webcam' && !file)} onClick={() => void start()}>{busy ? 'Đang xử lý…' : 'Bắt đầu'}</button>
        <button className="bg-red-700 text-white rounded px-4 py-2 disabled:opacity-50" disabled={!active || busy} onClick={() => void stop()}>Dừng nguồn</button>
        <button className={inputStyle} onClick={() => void enableSound()}>{sound ? 'Tắt âm cảnh báo' : 'Bật âm cảnh báo'}</button>
        {session?.download_available && <a className={inputStyle} href={`${API_URL}/download_processed_video/${session.session_id}`}>Tải video kết quả</a>}
      </div>
      <p className="text-sm text-slate-600">Trạng thái: {session?.status || 'Chưa chạy'} · Sự kiện: {connected ? 'Đã kết nối' : active ? 'Đang kết nối lại' : 'Đã dừng'}</p>
      {error && <p role="alert" className="text-red-700">{error}</p>}
    </div>
    <div className="grid lg:grid-cols-3 gap-5">
      <div className="lg:col-span-2 bg-slate-900 rounded-xl min-h-64 flex items-center justify-center overflow-hidden">
        {image ? <img src={image} alt="Kết quả nhận diện ảnh" className="w-full" /> : session ?
          <img key={session.session_id} src={`${API_URL}/realtime_video_stream/${session.session_id}`} alt="Luồng nhận diện AI" className="w-full" /> :
          <p className="text-slate-300 p-8">Chọn video, webcam hoặc ảnh để bắt đầu.</p>}
      </div>
      <div className="bg-white rounded-xl p-5 shadow space-y-3">
        <h3 className="font-bold">Trạng thái từng người</h3>
        {image && <p className="text-sm text-slate-600">Ảnh tĩnh: chỉ phân loại, không xác nhận cảnh báo theo thời gian.</p>}
        {!people.length && <p>Chưa có người được theo dõi; chưa thể kết luận an toàn.</p>}
        {people.map(person => <div key={person.track_id} className={`rounded p-3 border ${person.alert_level === 'drowning_alert' ? 'bg-red-100 border-red-500' : person.alert_level === 'struggling_alert' ? 'bg-yellow-100 border-yellow-500' : 'bg-slate-50'}`}>
          <p className="font-semibold">ID {person.track_id} · {classNames[person.class] || person.class}</p>
          <p>{person.alert_level ? levelNames[person.alert_level] || person.alert_level : 'Kết quả ảnh tĩnh'}</p>
          <p className="text-sm">Confidence: {person.confidence === null ? '—' : `${(person.confidence * 100).toFixed(1)}%`}</p>
        </div>)}
      </div>
    </div>
    {metrics && <div className="bg-white p-5 rounded-xl shadow grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
      {Object.entries({ 'FPS thực': metrics.fps, 'Xử lý (ms)': metrics.processing_ms, 'p50 (ms)': metrics.latency_p50_ms,
        'p95 (ms)': metrics.latency_p95_ms, 'Đọc (ms)': metrics.read_ms, 'GAN (ms)': metrics.gan_ms,
        'YOLO (ms)': metrics.yolo_ms, 'Tracking (ms)': metrics.tracking_ms, 'JPEG (ms)': metrics.jpeg_ms,
        'RAM tiến trình (MB)': metrics.rss_mb, 'CPU tiến trình (%)': metrics.cpu_percent }).map(([label, value]) => <p key={label}>{label}: <strong>{value ?? '—'}</strong></p>)}
    </div>}
  </div>;
}
