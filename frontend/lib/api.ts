export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const TOKEN_KEY = "meta_token";
export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}
export function setToken(token: string | null) {
  if (typeof window === "undefined") return;
  if (token) window.localStorage.setItem(TOKEN_KEY, token);
  else window.localStorage.removeItem(TOKEN_KEY);
}

async function req(path: string, init?: RequestInit): Promise<Response> {
  const headers: Record<string, string> = { ...(init?.headers as Record<string, string> || {}) };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (init?.body !== undefined && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  const r = await fetch(`${API}${path}`, { ...init, headers });
  if (r.status === 401 && typeof window !== "undefined") {
    setToken(null);
    if (window.location.pathname !== "/login") window.location.href = "/login";
  }
  return r;
}

export type Conversation = { id: string; title: string; model: string; created_at: string; updated_at: string };
export type Message = { id: string; conversation_id: string; role: string; content: string; created_at: string };

export async function listConversations(): Promise<Conversation[]> {
  const r = await req(`/api/conversations`);
  if (!r.ok) throw new Error("Failed to load conversations");
  return r.json();
}
export async function createConversation(): Promise<Conversation> {
  const r = await req(`/api/conversations`, {
    method: "POST", body: JSON.stringify({ title: "New chat" }),
  });
  if (!r.ok) throw new Error("Failed to create conversation");
  return r.json();
}
export async function getConversation(id: string) {
  const r = await req(`/api/conversations/${id}`);
  if (!r.ok) throw new Error("Failed to load conversation");
  return r.json() as Promise<Conversation & { messages: Message[] }>;
}
export async function renameConversation(id: string, title: string) {
  await req(`/api/conversations/${id}`, {
    method: "PATCH", body: JSON.stringify({ title }),
  });
}
export async function deleteConversation(id: string) {
  await req(`/api/conversations/${id}`, { method: "DELETE" });
}

// --- Auth --------------------------------------------------------------------

export type User = { id: string; username: string; trainings_used: number; max_trainings: number };
export async function register(username: string, password: string): Promise<{ token: string; user: User }> {
  const r = await req(`/api/auth/register`, {
    method: "POST", body: JSON.stringify({ username, password }),
  });
  if (!r.ok) {
    let msg = "Registration failed";
    try { msg = (await r.json())?.detail || msg; } catch { /* keep default */ }
    throw new Error(msg);
  }
  return r.json();
}
export async function login(username: string, password: string): Promise<{ token: string; user: User }> {
  const r = await req(`/api/auth/login`, {
    method: "POST", body: JSON.stringify({ username, password }),
  });
  if (!r.ok) throw new Error("Wrong username or password");
  return r.json();
}
export async function logout(): Promise<void> {
  try { await req(`/api/auth/logout`, { method: "POST" }); } catch { /* ignore */ }
  setToken(null);
}
export async function getMe(): Promise<User> {
  const r = await req(`/api/auth/me`);
  if (!r.ok) throw new Error("Not logged in");
  return r.json();
}

// --- Asset & Story engine ---------------------------------------------------

export type AssetKind = "character" | "object" | "environment";
export type Asset = {
  id: string; kind: AssetKind; name: string; description: string;
  dna: Record<string, string>; reference_image: string;
  reference_images: string[]; train_status: string; adapter_ref: string;
  train_call_id: string;
  version: number;
  created_at: string; updated_at: string;
};
export type Panel = {
  id: string; project_id: string; index: number; beat_title: string; action: string;
  dialogue: string; narration: string; camera: { shot?: string; angle?: string; lens?: string };
  layout: Record<string, unknown>; svg: string; image: string; prompt: string;
  consistency_score: number; issues: string[]; repaired: boolean; created_at: string;
};
export type Stage = {
  id: string; run_id: string; agent: string; order: number; status: string;
  summary: string; detail: Record<string, unknown>; created_at: string;
};
export type Run = {
  id: string; project_id: string; seed: number; status: string;
  bible: { assets?: Record<string, { name: string; kind: string; dna_hash: string }> };
  bible_hash: string; output_hash: string; consistency_avg: number;
  created_at: string; stages: Stage[];
};
export type Project = {
  id: string; title: string; premise: string; seed: number; panel_count: number;
  asset_ids: string[]; profile: string; status: string; created_at: string; updated_at: string;
};
export type ProjectDetail = Project & { panels: Panel[]; runs: Run[] };

async function ok<T>(r: Response, what: string): Promise<T> {
  if (!r.ok) {
    let msg = what;
    try { msg = (await r.json())?.detail || msg; } catch { /* keep default */ }
    throw new Error(msg);
  }
  return r.json();
}

export type EncodeResult = {
  dna: Record<string, string>; description: string; source: string; model: string;
  fallback: boolean; fallback_reason: string; palette: string[];
};
export async function encodeReference(image: string, kind: AssetKind): Promise<EncodeResult> {
  return ok(await req(`/api/assets/encode`, { method: "POST", body: JSON.stringify({ image, kind }) }), "Failed to read reference image");
}
export type FormatProfile = "tshirt" | "storyboard" | "video";
export type Graphic = {
  image: string; prompt: string; model: string; source: string;
  profile: string; checks: string[];
  bible_hash: string; output_hash: string; seed: number;
};
export async function renderGraphic(id: string, body?: { seed?: number; adapter?: boolean; lora_scale?: number; pose?: string; expression?: string; shot?: string; include_ids?: string[]; profile?: FormatProfile; action?: string; composition?: string; elements?: string; phrase?: string; phrase_placement?: string }): Promise<Graphic> {
  return ok(await req(`/api/assets/${id}/render`, { method: "POST", body: JSON.stringify(body || {}) }), "Failed to generate graphic");
}
export async function listAssets(): Promise<Asset[]> {
  return ok(await req(`/api/assets`), "Failed to load assets");
}
export type DatasetInfo = {
  trigger: string; count: number; recommended_minimum: number; ready: boolean;
  train_status: string; adapter_ref: string; train_call_id: string;
  captions: string[]; fingerprints: string[]; custom: boolean[];
};
export async function getDataset(id: string): Promise<DatasetInfo> {
  return ok(await req(`/api/assets/${id}/dataset`), "Failed to load dataset");
}
export type GradeCheck = { level: string; code: string; message: string };
export type GradeInfo = {
  ready: boolean; count: number; distinct: number; recommended_minimum: number;
  trigger: string; checks: GradeCheck[];
};
export async function getGrade(id: string): Promise<GradeInfo> {
  return ok(await req(`/api/assets/${id}/grade`), "Failed to grade dataset");
}
export async function saveCaptions(id: string, overrides: Record<string, string>): Promise<{ updated: number }> {
  return ok(await req(`/api/assets/${id}/captions`, { method: "PATCH", body: JSON.stringify({ overrides }) }), "Failed to save captions");
}
export async function trainAsset(id: string, steps?: number): Promise<{ status: string; message?: string; call_id?: string }> {
  return ok(await req(`/api/assets/${id}/train`, { method: "POST", body: JSON.stringify({ steps: steps ?? null }) }), "Failed to start training");
}
export type ValidationProbe = { name: string; passed: boolean; reasons: string[] };
export type ValidationResult = {
  passed: boolean; adapter_ref: string; trigger: string; probes: ValidationProbe[];
};
export type TrainingStatus = {
  status: string; adapter_ref?: string; message?: string; call_id?: string;
  validation?: ValidationResult | Record<string, never>;
};
export async function getTrainingStatus(id: string): Promise<TrainingStatus> {
  return ok(await req(`/api/assets/${id}/train/status`), "Failed to check training status");
}
export async function validateAdapter(id: string): Promise<TrainingStatus> {
  return ok(await req(`/api/assets/${id}/validate`, { method: "POST" }), "Failed to validate adapter");
}
export async function createAsset(body: { kind: AssetKind; name: string; description: string; dna: Record<string, string>; reference_image?: string; reference_images?: string[] }): Promise<Asset> {
  return ok(await req(`/api/assets`, { method: "POST", body: JSON.stringify(body) }), "Failed to create asset");
}
export async function updateAsset(id: string, body: Partial<Asset>): Promise<Asset> {
  return ok(await req(`/api/assets/${id}`, { method: "PATCH", body: JSON.stringify(body) }), "Failed to update asset");
}
export async function deleteAsset(id: string): Promise<void> {
  const r = await req(`/api/assets/${id}`, { method: "DELETE" });
  if (!r.ok) throw new Error("Failed to delete asset");
}
export async function listProjects(): Promise<Project[]> {
  return ok(await req(`/api/projects`), "Failed to load projects");
}
export async function createProject(body: { title: string; premise: string; seed: number; panel_count: number; asset_ids: string[]; profile?: string }): Promise<Project> {
  return ok(await req(`/api/projects`, { method: "POST", body: JSON.stringify(body) }), "Failed to create project");
}
export async function getProject(id: string): Promise<ProjectDetail> {
  return ok(await req(`/api/projects/${id}`), "Failed to load project");
}
export async function updateProject(id: string, body: Partial<Project>): Promise<Project> {
  return ok(await req(`/api/projects/${id}`, { method: "PATCH", body: JSON.stringify(body) }), "Failed to update project");
}
export async function deleteProject(id: string): Promise<void> {
  const r = await req(`/api/projects/${id}`, { method: "DELETE" });
  if (!r.ok) throw new Error("Failed to delete project");
}
export async function runProject(id: string, seed?: number, panelCount?: number, profile?: FormatProfile): Promise<Run> {
  return ok(await req(`/api/projects/${id}/run`, {
    method: "POST",
    body: JSON.stringify({ seed: seed ?? null, panel_count: panelCount ?? null, profile: profile ?? "storyboard" }),
  }), "Pipeline run failed");
}
export async function exportProject(id: string): Promise<unknown> {
  return ok(await req(`/api/projects/${id}/export`), "Export failed");
}
export async function seedDemo(): Promise<{ project_id: string }> {
  return ok(await req(`/api/demo`, { method: "POST" }), "Demo failed");
}

export async function streamMessage(
  conversationId: string, content: string,
  onDelta: (text: string) => void, onMeta: (m: { model?: string }) => void,
  onError?: (message: string) => void,
): Promise<string> {
  const res = await req(`/api/conversations/${conversationId}/messages`, {
    method: "POST", body: JSON.stringify({ content }),
  });
  if (!res.ok || !res.body) throw new Error("Failed to send message");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", full = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      try {
        const evt = JSON.parse(line.slice(5).trim());
        if (evt.type === "delta") { full += evt.content; onDelta(evt.content); }
        if (evt.type === "meta") onMeta(evt);
        if (evt.type === "error") { onError?.(evt.message || "META API error"); throw new Error(evt.message || "META API error"); }
        if (evt.type === "done") full = evt.content ?? full;
      } catch { /* keepalive */ }
    }
  }
  return full;
}
