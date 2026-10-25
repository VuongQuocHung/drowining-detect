export const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
export interface PersonDetection {
  track_id: number; class: string; confidence: number | null;
  alert_level?: string; previous_level?: string; alarm?: boolean;
}
export interface SessionInfo {
  session_id: string; model_type: string; source: 'webcam' | 'video'; status: string;
  error?: string | null; processing_complete: boolean; download_available: boolean;
  enhancement: boolean; confidence_threshold: number;
  frame_count?: number;
}
export interface SessionEvent {
  sequence: number; session_id: string; status: string; source_time?: number;
  detections: PersonDetection[]; metrics?: Record<string, number>; terminal?: boolean; error?: string | null;
}
export async function sessionRequest<T>(path: string, body?: FormData): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, body ? { method: 'POST', body } : undefined);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as { detail?: string };
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}
export function stopSession(id: string): Promise<SessionInfo> {
  return sessionRequest(`/stop_video_processing/${id}`, new FormData());
}
