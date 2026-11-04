"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import TrainPanel from "@/components/TrainPanel";
import { Asset, AssetKind, createAsset, deleteAsset, encodeReference, listAssets, updateAsset } from "@/lib/api";

type Field = { key: string; label: string; type: "color" | "text" | "select"; options?: string[] };
const FIELDS: Record<AssetKind, Field[]> = {
  character: [
    { key: "form", label: "Form", type: "select", options: ["human", "fox", "cat", "creature"] },
    { key: "build", label: "Build", type: "select", options: ["slim", "average", "broad"] },
    { key: "skin", label: "Skin / fur base", type: "color" },
    { key: "hair", label: "Hair", type: "color" },
    { key: "hair_style", label: "Hair style", type: "select", options: ["short", "long", "bald"] },
    { key: "facial_hair", label: "Facial hair", type: "select", options: ["none", "beard", "moustache"] },
    { key: "headwear_shape", label: "Headwear shape", type: "select", options: ["auto", "none", "brim", "pointed", "hood"] },
    { key: "ear_shape", label: "Ear shape", type: "select", options: ["auto", "none", "pointed", "round", "animal"] },
    { key: "eyes", label: "Eyes", type: "color" },
    { key: "outfit_top", label: "Outfit top", type: "color" },
    { key: "outfit_bottom", label: "Outfit bottom", type: "color" },
    { key: "accent", label: "Accent", type: "color" },
    { key: "accessory", label: "Signature accessory", type: "text" },
    { key: "marking", label: "Marking", type: "text" },
  ],
  object: [
    { key: "shape", label: "Shape", type: "select", options: ["orb", "crystal", "box", "book", "sword", "key"] },
    { key: "material", label: "Material", type: "text" },
    { key: "primary", label: "Primary", type: "color" },
    { key: "secondary", label: "Secondary", type: "color" },
    { key: "accent", label: "Accent", type: "color" },
    { key: "marking", label: "Marking", type: "text" },
    { key: "glow", label: "Glow", type: "select", options: ["none", "soft", "bright"] },
  ],
  environment: [
    { key: "biome", label: "Biome", type: "text" },
    { key: "time", label: "Time", type: "select", options: ["day", "dusk", "night"] },
    { key: "sky", label: "Sky", type: "color" },
    { key: "ground", label: "Ground", type: "color" },
    { key: "primary", label: "Primary", type: "color" },
    { key: "secondary", label: "Silhouettes", type: "color" },
    { key: "accent", label: "Accent", type: "color" },
    { key: "landmark", label: "Landmark", type: "select", options: ["towers", "mountains", "forest"] },
    { key: "weather", label: "Weather", type: "text" },
  ],
};
const STARTER: Record<AssetKind, Record<string, string>> = {
  character: { form: "human", build: "average", skin: "#e8b08a", hair: "#3b2a20", hair_style: "short", facial_hair: "none", headwear_shape: "auto", ear_shape: "auto", eyes: "#2b2b2b", outfit_top: "#256d5a", outfit_bottom: "#33404f", accent: "#d9a03c", accessory: "none", marking: "none" },
  object: { shape: "orb", material: "brass", primary: "#b08a3c", secondary: "#6e5426", accent: "#ffe9a8", marking: "none", glow: "none" },
  environment: { biome: "city", time: "day", sky: "#a8c8e0", ground: "#7a8a5a", primary: "#5a7a9a", secondary: "#3c4c5e", accent: "#d9a03c", landmark: "towers", weather: "clear" },
};
const KIND_LABEL: Record<AssetKind, string> = { character: "Character", object: "Object", environment: "Environment" };

export default function AssetsPage() {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [error, setError] = useState("");
  const [kind, setKind] = useState<AssetKind>("character");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [dna, setDna] = useState<Record<string, string>>(STARTER.character);
  const [refImages, setRefImages] = useState<string[]>([]);
  const refImage = refImages[0] || "";
  const [encoding, setEncoding] = useState(false);
  const [encodeNote, setEncodeNote] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try { setAssets(await listAssets()); setError(""); }
    catch { setError("Backend unreachable — is FastAPI running on :8000?"); }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  function switchKind(k: AssetKind) { setKind(k); setDna(STARTER[k]); setEditing(null); setEncodeNote(""); }
  // Shrink a copy for the vision call only; the stored reference keeps
  // its original resolution.
  function scaledForEncode(dataUrl: string): Promise<string> {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, 768 / Math.max(img.width, img.height));
        if (scale >= 1) { resolve(dataUrl); return; }
        const c = document.createElement("canvas");
        c.width = Math.round(img.width * scale); c.height = Math.round(img.height * scale);
        c.getContext("2d")?.drawImage(img, 0, 0, c.width, c.height);
        resolve(c.toDataURL("image/jpeg", 0.85));
      };
      img.onerror = () => resolve(dataUrl);
      img.src = dataUrl;
    });
  }
  async function onFiles(files: FileList | null) {
    if (!files?.length) return;
    for (const f of Array.from(files)) {
      const full = await new Promise<string>((resolve) => {
        const r = new FileReader();
        r.onload = () => resolve(String(r.result));
        r.readAsDataURL(f);
      });
      const first = refImages.length === 0;
      setRefImages((prev) => [...prev, full]);
      if (first) await applyEncode(full);
    }
  }
  async function applyEncode(dataUrl: string) {
    setEncoding(true); setEncodeNote("Reading your image to fill the DNA…");
    try {
      const res = await encodeReference(await scaledForEncode(dataUrl), kind);
      setDna((prev) => ({ ...prev, ...res.dna }));
      if (res.description) setDescription((prev) => prev.trim() ? prev : res.description);
      setEncodeNote(res.fallback
        ? "Vision was unavailable, so colours were guessed from the image palette. Check every field, then save."
        : `DNA auto-filled from your image by ${res.model || "the Meta model"} — check it, then save.`);
    } catch {
      setEncodeNote("Could not read the image automatically — fill in the DNA manually below.");
    } finally { setEncoding(false); }
  }
  function startEdit(a: Asset) {
    setEditing(a.id); setKind(a.kind); setName(a.name); setDescription(a.description);
    setDna({ ...STARTER[a.kind], ...a.dna }); setRefImages(a.reference_images?.length ? a.reference_images : (a.reference_image ? [a.reference_image] : [])); setEncodeNote("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  async function onSave(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim() || busy) return;
    setBusy(true); setError("");
    try {
      if (editing) await updateAsset(editing, { name: name.trim(), description, dna, reference_image: refImage, reference_images: refImages });
      else await createAsset({ kind, name: name.trim(), description, dna, reference_image: refImage, reference_images: refImages });
      setName(""); setDescription(""); setRefImages([]); setEncodeNote(""); setEditing(null); setDna(STARTER[kind]);
      await refresh();
    } catch (err) { setError(err instanceof Error ? err.message : "Save failed"); }
    finally { setBusy(false); }
  }
  async function onDelete(a: Asset) {
    if (!window.confirm(`Delete asset "${a.name}"? Stories keep their frozen Bible.`)) return;
    await deleteAsset(a.id); refresh();
  }

  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-6">
      <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm">
      <span className="inline-block rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-semibold text-amber-900">
        Source of truth
      </span>
      <h1 className="mt-2 text-3xl font-bold tracking-tight">Asset Bible</h1>
      <p className="mt-1 max-w-xl text-sm leading-relaxed text-zinc-600">
        DNA is the source of truth. When a story runs, the Canon Keeper freezes these values into a
        hashed Bible — editing an asset later never changes a finished board.
      </p>
      </div>
      {error && <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2.5 text-sm text-red-800" role="alert">{error}</div>}

      <section className="mt-6 rounded-xl border border-zinc-200 bg-white p-5 shadow-sm">
        <h2 className="text-base font-semibold">{editing ? "Edit asset" : "Define an asset"}</h2>
        <div className="mt-3 flex gap-2" role="tablist" aria-label="Asset kind">
          {(Object.keys(KIND_LABEL) as AssetKind[]).map((k) => (
            <button key={k} onClick={() => switchKind(k)} disabled={!!editing}
              className={`min-h-11 rounded-full border px-4 text-sm font-medium transition-colors disabled:opacity-40 ${kind === k ? "border-zinc-900 bg-zinc-900 text-white" : "border-zinc-300 bg-white hover:border-zinc-400 hover:bg-zinc-50"}`}>
              {KIND_LABEL[k]}
            </button>
          ))}
        </div>
        <form onSubmit={onSave} className="mt-4 grid gap-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name (e.g. Mira Vale)"
              className="min-h-11 rounded-lg border border-zinc-300 px-3 text-[15px] outline-none placeholder:text-zinc-400 focus:border-zinc-500" />
            <input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Short description"
              className="min-h-11 rounded-lg border border-zinc-300 px-3 text-[15px] outline-none placeholder:text-zinc-400 focus:border-zinc-500" />
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {FIELDS[kind].map((f) => (
              <label key={f.key} className="text-xs font-medium text-zinc-600">{f.label}
                {f.type === "color" ? (
                  <span className="mt-1 flex min-h-11 items-center gap-2 rounded-lg border border-zinc-300 bg-white px-2">
                    <input type="color" value={dna[f.key] || "#888888"} onChange={(e) => setDna({ ...dna, [f.key]: e.target.value })}
                      className="h-8 w-10 shrink-0 cursor-pointer" />
                    <span className="font-mono text-[11px]">{dna[f.key] || ""}</span>
                  </span>
                ) : f.type === "select" ? (
                  <select value={dna[f.key] || f.options?.[0]} onChange={(e) => setDna({ ...dna, [f.key]: e.target.value })}
                    className="mt-1 min-h-11 w-full rounded-lg border border-zinc-300 bg-white px-2 text-sm text-zinc-900">
                    {f.options?.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                ) : (
                  <input value={dna[f.key] || ""} onChange={(e) => setDna({ ...dna, [f.key]: e.target.value })}
                    className="mt-1 min-h-11 w-full rounded-lg border border-zinc-300 bg-white px-2 text-sm text-zinc-900" />
                )}
              </label>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <input ref={fileRef} type="file" accept="image/*" multiple className="hidden" onChange={(e) => onFiles(e.target.files)} />
            <button type="button" onClick={() => fileRef.current?.click()}
              className="min-h-11 rounded-full border border-zinc-300 bg-white px-4 text-sm font-medium transition-colors hover:border-zinc-400 hover:bg-zinc-50">
              Upload turnaround / reference
            </button>
            {refImage && (
              <>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={refImage} alt="Reference" className="h-14 w-14 rounded-lg border border-zinc-200 object-cover" />
                <button type="button" onClick={() => { setRefImages([]); setEncodeNote(""); }} className="text-sm text-zinc-500 hover:text-red-700">Remove</button>
                <button type="button" disabled={encoding} onClick={() => applyEncode(refImage)} className="text-sm font-medium text-teal-800 underline disabled:opacity-40">Re-read image into DNA</button>
              </>
            )}
            <span className="text-xs text-zinc-500">Upload one or more references (more = better LoRA dataset). The first auto-fills the DNA below — review it before saving.</span>
          </div>
          {refImages.length > 1 && (
            <div className="flex flex-wrap gap-2">
              {refImages.map((src, i) => (
                <span key={i} className="relative inline-block">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={src} alt={`Reference ${i + 1}`} className="h-14 w-14 rounded-lg border border-zinc-200 object-cover" />
                  <button type="button" aria-label={`Remove image ${i + 1}`}
                    onClick={() => setRefImages((prev) => prev.filter((_, j) => j !== i))}
                    className="absolute -right-2 -top-2 flex h-6 w-6 items-center justify-center rounded-full bg-zinc-900 text-xs text-white">×</button>
                </span>
              ))}
              <span className="text-xs text-zinc-500">{refImages.length} training images</span>
            </div>
          )}
          {encodeNote && <p className="text-[13px] text-teal-800" role="status">{encodeNote}</p>}
          <div className="flex gap-3">
            <button type="submit" disabled={!name.trim() || busy || encoding}
              className="min-h-11 rounded-full bg-zinc-900 px-6 text-sm font-semibold text-white transition-colors hover:bg-zinc-700 disabled:opacity-40">
              {encoding ? "Reading image…" : editing ? "Save new version" : "Add to Bible"}
            </button>
            {editing && <button type="button" onClick={() => { setEditing(null); setName(""); setDescription(""); setRefImages([]); }} className="min-h-11 px-3 text-sm text-zinc-600">Cancel</button>}
          </div>
        </form>
      </section>

      <p className="mt-8 text-sm text-zinc-700">
        This Bible is your library. To make something with a figure, head to{" "}
        <a href="/" className="font-medium text-teal-800 underline">Create</a> and pick the purpose and variation there.
      </p>

      <section className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {assets.map((a) => (
          <article key={a.id} className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm">
            <div className="flex items-start justify-between gap-2">
              <h3 className="text-[15px] font-semibold">{a.name}</h3>
              <span className="rounded-full bg-zinc-900 px-2 py-0.5 font-mono text-[10px] font-bold text-white">{a.kind} · v{a.version}</span>
            </div>
            <p className="mt-1 text-sm leading-relaxed text-zinc-700">{a.description || "—"}</p>
            {a.reference_image && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={a.reference_image} alt={`${a.name} turnaround`} className="mt-2 h-28 w-full rounded-lg border border-zinc-200 object-cover" />
            )}
            <div className="mt-2 flex flex-wrap gap-1">
              {Object.entries(a.dna).filter(([, v]) => String(v).startsWith("#")).map(([k, v]) => (
                <span key={k} title={`${k}: ${v}`} className="inline-block h-5 w-5 rounded-full border border-zinc-200" style={{ background: v }} />
              ))}
            </div>
            <p className="mt-2 font-mono text-[11px] leading-relaxed text-zinc-600">
              {Object.entries(a.dna).filter(([, v]) => !String(v).startsWith("#")).slice(0, 4).map(([k, v]) => `${k}: ${v}`).join(" · ")}
            </p>
            <div className="mt-3 flex gap-3 text-sm">
              <button onClick={() => startEdit(a)} className="font-medium text-teal-800 underline">Edit DNA</button>
              <button onClick={() => onDelete(a)} className="text-zinc-500 hover:text-red-700">Delete</button>
            </div>
            {a.kind === "character" && <TrainPanel asset={a} onTrained={refresh} />}
          </article>
        ))}
        {assets.length === 0 && <p className="text-sm text-zinc-600">No assets yet. Define your first character above.</p>}
      </section>
    </main>
  );
}
