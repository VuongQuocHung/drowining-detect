// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import DrowningDetection from './DrowningDetection';
import { SessionEvent, SessionInfo, sessionRequest } from '../services/sessions';

vi.mock('../services/sessions', () => ({
  API_URL: 'http://test', sessionRequest: vi.fn(), stopSession: vi.fn(),
}));
const request = vi.mocked(sessionRequest);
const info: SessionInfo = { session_id: 'one', model_type: 'underwater', source: 'video', status: 'running',
  processing_complete: false, download_available: false, enhancement: false, confidence_threshold: .5 };
class FakeEvents {
  static instances: FakeEvents[] = [];
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  close = vi.fn();
  constructor(public url: string) { FakeEvents.instances.push(this); }
  emit(event: SessionEvent) { this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(event) })); }
}
const beep = vi.fn();
class FakeAudio {
  state = 'running'; currentTime = 0; destination = {};
  resume = vi.fn().mockResolvedValue(undefined);
  close = vi.fn().mockResolvedValue(undefined);
  createOscillator() { return { frequency: { value: 0 }, connect: vi.fn(), start: beep, stop: vi.fn() }; }
  createGain() { return { gain: { value: 0 }, connect: vi.fn() }; }
}
function event(sequence: number, level = 'drowning_alert', alarm = true): SessionEvent {
  return { sequence, session_id: 'one', status: 'running', source_time: sequence,
    detections: [{ track_id: 1, class: 'drowning', confidence: .9, alert_level: level, alarm }] };
}
beforeEach(() => {
  FakeEvents.instances = []; beep.mockReset(); request.mockReset();
  request.mockResolvedValue(null);
  vi.stubGlobal('EventSource', FakeEvents); vi.stubGlobal('AudioContext', FakeAudio);
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('AI session dashboard', () => {
  it('restores the active session without starting a second worker', async () => {
    request.mockResolvedValue(info);
    render(<DrowningDetection />);
    await waitFor(() => expect(FakeEvents.instances).toHaveLength(1));
    expect(request).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Bắt đầu' }).hasAttribute('disabled')).toBe(true);
    act(() => FakeEvents.instances[0].emit(event(1)));
    expect(screen.getByText('Cảnh báo đuối nước')).toBeTruthy();
    expect(beep).not.toHaveBeenCalled();
  });

  it('requires a sound gesture, ignores yellow alerts and deduplicates red pulses', async () => {
    request.mockResolvedValue(info);
    render(<DrowningDetection />);
    await waitFor(() => expect(FakeEvents.instances).toHaveLength(1));
    const source = FakeEvents.instances[0];
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Bật âm cảnh báo' })); });
    act(() => source.emit(event(1, 'struggling_alert', false)));
    expect(beep).not.toHaveBeenCalled();
    act(() => source.emit(event(2)));
    expect(beep).toHaveBeenCalledTimes(1);
    act(() => source.emit(event(2)));
    expect(beep).toHaveBeenCalledTimes(1);
    act(() => source.emit(event(3)));
    expect(beep).toHaveBeenCalledTimes(2);
    act(() => source.emit({ ...event(4), status: 'failed', terminal: true, error: 'Camera disconnected', detections: [] }));
    expect(source.close).toHaveBeenCalled();
    expect(screen.getByRole('alert').textContent).toContain('Camera disconnected');
  });

  it('hides lost tracks unless they were alerting', async () => {
    request.mockResolvedValue(info);
    render(<DrowningDetection />);
    await waitFor(() => expect(FakeEvents.instances).toHaveLength(1));
    const lost = (track_id: number, previous_level: string) =>
      ({ track_id, class: 'unknown', confidence: null, alert_level: 'lost', previous_level, alarm: false });
    act(() => FakeEvents.instances[0].emit({ ...event(1, 'normal', false),
      detections: [{ track_id: 9, class: 'swimming', confidence: .8, alert_level: 'normal', alarm: false },
        lost(1, 'normal'), lost(2, 'suspected'), lost(3, 'drowning_alert')] }));
    expect(screen.getByText(/ID 9/)).toBeTruthy();
    expect(screen.getByText(/ID 3/)).toBeTruthy();
    expect(screen.queryByText(/ID 1 /)).toBeNull();
    expect(screen.queryByText(/ID 2 /)).toBeNull();
    expect(screen.getByText(/trước đó: Cảnh báo đuối nước/)).toBeTruthy();
    expect(screen.getByText(/2 ID vừa mất dấu/)).toBeTruthy();
  });

  it('uploads with GAN disabled by default and displays server conflict details', async () => {
    render(<DrowningDetection />);
    await waitFor(() => expect(request).toHaveBeenCalledTimes(1));
    request.mockRejectedValueOnce(new Error('Stop the active source before starting another'));
    fireEvent.change(screen.getByLabelText('Tệp đầu vào'), { target: { files: [new File(['video'], 'input.mp4', { type: 'video/mp4' })] } });
    fireEvent.click(screen.getByRole('button', { name: 'Bắt đầu' }));
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('Stop the active source'));
    const body = request.mock.calls[1][1] as FormData;
    expect(body.get('enhancement')).toBe('false');
    expect(body.get('confidence_threshold')).toBe('0.5');
  });

  it('treats still images as classifications without temporal alerts or SSE', async () => {
    render(<DrowningDetection />);
    await waitFor(() => expect(request).toHaveBeenCalledTimes(1));
    request.mockResolvedValueOnce({ image_base64: 'AA==', detections: [{ track_id: 1, class: 'struggling', confidence: .8 }] });
    fireEvent.change(screen.getByLabelText('Nguồn'), { target: { value: 'image' } });
    fireEvent.change(screen.getByLabelText('Tệp đầu vào'), { target: { files: [new File(['image'], 'image.jpg', { type: 'image/jpeg' })] } });
    fireEvent.click(screen.getByRole('button', { name: 'Bắt đầu' }));
    await waitFor(() => expect(screen.getByAltText('Kết quả nhận diện ảnh')).toBeTruthy());
    expect(FakeEvents.instances).toHaveLength(0);
    expect(beep).not.toHaveBeenCalled();
  });
});
