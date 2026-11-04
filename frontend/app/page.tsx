"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  Asset, FormatProfile, Graphic, Project, createProject, deleteProject,
  listAssets, listProjects, renderGraphic, runProject, seedDemo,
} from "@/lib/api";

const KIND_BADGE: Record<string, string> = {
  character: "bg-teal-800 text-teal-50",
  object: "bg-amber-500 text-zinc-900",
  environment: "bg-sky-800 text-sky-50",
};
const PROFILES: { id: FormatProfile; label: string; hint: string }[] = [
  { id: "tshirt", label: "T-shirt graphic", hint: "One isolated figure on transparency, contained with margins." },
  { id: "storyboard", label: "Storyboard", hint: "Full-bleed scene panels from a premise." },
  { id: "video", label: "Short video frames", hint: "16:9 keyframes with headroom for motion." },
];

export default function CreateHome() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [profile, setProfile] = useState<FormatProfile>("tshirt");
  const [leadId, setLeadId] = useState("");
  const [extras, setExtras] = useState<string[]>([]);
  const [action, setAction] = useState("");
  const [pose, setPose] = useState("stand");
  const [expression, setExpression] = useState("calm");
  const [shot, setShot] = useState("medium");
  const [seed, setSeed] = useState(42);
  const [count, setCount] = useState(3);
  const [composition, setComposition] = useState("vignette");
  const [elements, setElements] = useState("");
  const [phrase, setPhrase] = useState("");
  const [phrasePlacement, setPhrasePlacement] = useState("below");
  const [useAdapter, setUseAdapter] = useState(true);
  const [loraScale, setLoraScale] = useState(0.85);
  const [graphic, setGraphic] = useState<(Graphic & { name: string }) | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [p, a] = await Promise.all([listProjects(), listAssets()]);
      setProjects(p); setAssets(a); setError("");
    } catch { setError("Backend unreachable — is FastAPI running on :8000?"); }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  const characters = assets.filter((a) => a.kind === "character");
  const lead = assets.find((a) => a.id === leadId);
  const leadHasAdapter = Boolean(lead?.train_status === "ready" && lead?.adapter_ref.startsWith("modal://"));

  function toggleExtra(id: string) {
    setExtras((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));
  }

  async function onGenerate(e: React.FormEvent) {
    e.preventDefault();
    if (!lead || busy) return;
    setBusy(true); setError("");
    try {
      if (profile === "tshirt") {
        const g = await renderGraphic(lead.id, {
          seed: Number(seed) || 42, adapter: leadHasAdapter ? useAdapter : false,
          lora_scale: loraScale,
          profile, pose, expression, shot,
          action: action.trim(), include_ids: extras,
          composition, elements: elements.trim(),
          phrase: phrase.trim(), phrase_placement: phrasePlacement,
        });
        setGraphic({ ...g, name: lead.name });
        setBusy(false);
      } else {
        const p = await createProject({
          title: `${lead.name} — ${PROFILES.find((x) => x.id === profile)?.label}`,
          premise: action.trim(), seed: Number(seed) || 42,
          panel_count: count, asset_ids: [lead.id, ...extras], profile,
        });
        await runProject(p.id, Number(seed) || 42, count, profile);
        window.location.href = `/projects/${p.id}`;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
      setBusy(false);
    }
  }

  async function onDemo() {
    setBusy(true); setError("");
    try {
      const { project_id } = await seedDemo();
      window.location.href = `/projects/${project_id}`;
    } catch (err) { setError(err instanceof Error ? err.message : "Demo failed"); setBusy(false); }
  }

  async function onDelete(p: Project) {
    if (!window.confirm(`Delete "${p.title}"?`)) return;
    await deleteProject(p.id); refresh();
  }

  const byId = Object.fromEntries(assets.map((a) => [a.id, a]));
  const selectCls = "mt-1 min-h-11 w-full rounded-lg border border-zinc-300 bg-white px-2 text-sm text-zinc-900";
  const inputCls = "mt-1 min-h-11 w-full rounded-lg border border-zinc-300 bg-white px-2 text-sm text-zinc-900";

  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-8">
      <div className="flex flex-wrap items-end justify-between gap-4 rounded-xl border border-zinc-200 bg-white p-6 shadow-sm">
        <div>
          <span className="inline-block rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-semibold text-amber-900">
            Character series studio
          </span>
          <h1 className="mt-2 text-3xl font-bold tracking-tight">Create</h1>
          <p className="mt-1 max-w-xl text-sm leading-relaxed text-zinc-600">
            Pick a figure from the Asset Bible, say what it will be used for, dictate the
            variation, and generate. The Bible only stores figures — everything is made here.
          </p>
        </div>
        <button onClick={onDemo} disabled={busy}
          className="min-h-11 rounded-full border border-zinc-300 bg-white px-5 text-sm font-semibold text-zinc-800 transition-colors hover:border-zinc-400 hover:bg-zinc-50 disabled:opacity-40">
          Load demo story
        </button>
      </div>

      {error && <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2.5 text-sm text-red-800" role="alert">{error}</div>}

      <form onSubmit={onGenerate} className="mt-6 grid gap-5">
        <section className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm">
          <h2 className="flex items-center gap-2 text-base font-semibold"><span className="flex h-6 w-6 items-center justify-center rounded-full bg-zinc-100 text-xs font-bold text-zinc-600">1</span> Figure</h2>
          {characters.length === 0
            ? <p className="mt-2 text-sm text-zinc-600">No characters yet — <Link href="/assets" className="font-medium text-teal-800 underline">define one in the Asset Bible</Link> or load the demo.</p>
            : <div className="mt-3 flex flex-wrap gap-2">
                {characters.map((a) => (
                  <button type="button" key={a.id} onClick={() => setLeadId(a.id)} aria-pressed={leadId === a.id}
                    className={`flex min-h-11 items-center gap-2 rounded-full border px-4 text-sm font-medium transition-colors ${leadId === a.id ? "border-zinc-900 bg-zinc-900 text-white" : "border-zinc-300 bg-white hover:border-zinc-400 hover:bg-zinc-50"}`}>
                    {a.reference_image && <img src={a.reference_image} alt="" className="h-8 w-8 rounded-full object-cover" />}
                    {a.name}
                  </button>
                ))}
              </div>}
        </section>

        <section className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm">
          <h2 className="flex items-center gap-2 text-base font-semibold"><span className="flex h-6 w-6 items-center justify-center rounded-full bg-zinc-100 text-xs font-bold text-zinc-600">2</span> Purpose</h2>
          <div className="mt-3 flex flex-wrap gap-2" role="tablist" aria-label="Purpose">
            {PROFILES.map((p) => (
              <button type="button" key={p.id} onClick={() => setProfile(p.id)} aria-pressed={profile === p.id}
                className={`min-h-11 rounded-full border px-4 text-sm font-medium transition-colors ${profile === p.id ? "border-zinc-900 bg-zinc-900 text-white" : "border-zinc-300 bg-white hover:border-zinc-400 hover:bg-zinc-50"}`}>
                {p.label}
              </button>
            ))}
          </div>
          <p className="mt-2 text-sm text-zinc-700">{PROFILES.find((p) => p.id === profile)?.hint}</p>
          {profile === "tshirt" && leadHasAdapter && (
            <div className="mt-3 space-y-2">
              <label className="flex items-start gap-2 text-sm text-zinc-700">
                <input type="checkbox" checked={useAdapter} onChange={(e) => setUseAdapter(e.target.checked)} className="mt-1" />
                <span>Generate through the trained Modal LoRA instead of the Meta reference model. Adapter inference uses paid Modal GPU time.</span>
              </label>
              {useAdapter && (
                <label className="flex max-w-xs items-center gap-2 text-sm text-zinc-700">
                  <span className="shrink-0">LoRA strength</span>
                  <input type="range" min={0.1} max={1} step={0.05} value={loraScale}
                    onChange={(e) => setLoraScale(Number(e.target.value))} className="w-full" />
                  <span className="w-10 shrink-0 font-mono text-xs">{loraScale.toFixed(2)}</span>
                </label>
              )}
            </div>
          )}
          {profile === "tshirt" && lead && !leadHasAdapter && (
            <p className="mt-3 text-sm text-zinc-600">No ready LoRA for {lead.name} yet — this will use Meta reference/likeness generation. Train one in the Asset Bible for adapter generation.</p>
          )}
        </section>

        <section className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm">
          <h2 className="flex items-center gap-2 text-base font-semibold"><span className="flex h-6 w-6 items-center justify-center rounded-full bg-zinc-100 text-xs font-bold text-zinc-600">3</span> Variation</h2>
          <textarea value={action} onChange={(e) => setAction(e.target.value)} rows={2}
            placeholder={profile === "tshirt"
              ? "What are they doing? e.g. holding a watering can, waving"
              : "Premise / scene — e.g. Norman weeds the garden and finds a lost key"}
            className="mt-3 w-full rounded-lg border border-zinc-300 bg-white px-3 py-2.5 text-[15px] outline-none placeholder:text-zinc-400 focus:border-zinc-500" />
          {profile === "tshirt" && (
            <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <label className="text-xs font-medium text-zinc-600">Pose
                <select value={pose} onChange={(e) => setPose(e.target.value)} className={selectCls}>
                  <option value="stand">Standing</option><option value="walk">Walking</option>
                  <option value="reach">Reaching</option><option value="leap">Leaping</option>
                </select></label>
              <label className="text-xs font-medium text-zinc-600">Expression
                <select value={expression} onChange={(e) => setExpression(e.target.value)} className={selectCls}>
                  <option value="calm">Calm</option><option value="curious">Curious</option>
                  <option value="surprised">Surprised</option><option value="determined">Determined</option>
                  <option value="relieved">Relieved</option>
                </select></label>
              <label className="text-xs font-medium text-zinc-600">Camera
                <select value={shot} onChange={(e) => setShot(e.target.value)} className={selectCls}>
                  <option value="wide">Wide</option><option value="medium">Medium</option>
                  <option value="close">Close-up</option>
                </select></label>
              <label className="text-xs font-medium text-zinc-600">Seed
                <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))}
                  className={inputCls + " font-mono"} /></label>
              <label className="text-xs font-medium text-zinc-600">Design composition
                <select value={composition} onChange={(e) => setComposition(e.target.value)} className={selectCls}>
                  <option value="figure">Figure only</option>
                  <option value="vignette">Figure + scene vignette</option>
                  <option value="badge">Circular badge</option>
                </select></label>
              <label className="text-xs font-medium text-zinc-600">Decorative elements
                <input value={elements} onChange={(e) => setElements(e.target.value)}
                  placeholder="mushrooms, sun, flowers"
                  className={inputCls} /></label>
              <label className="text-xs font-medium text-zinc-600">Phrase (optional, spelled exactly)
                <input value={phrase} onChange={(e) => setPhrase(e.target.value)}
                  placeholder="GNOME SWEET GNOME"
                  className={inputCls} /></label>
              <label className="text-xs font-medium text-zinc-600">Phrase placement
                <select value={phrasePlacement} onChange={(e) => setPhrasePlacement(e.target.value)} className={selectCls}>
                  <option value="below">Below design</option>
                  <option value="above">Above design</option>
                  <option value="banner">On a banner</option>
                </select></label>
            </div>
          )}
          {profile !== "tshirt" && (
            <div className="mt-3 flex flex-wrap gap-3">
              <label className="text-sm text-zinc-600">{profile === "video" ? "Frames" : "Panels"}
                <input type="number" min={1} max={12} value={count} onChange={(e) => setCount(Number(e.target.value))}
                  className="ml-2 min-h-11 w-20 rounded-lg border border-zinc-300 px-2 font-mono text-sm" /></label>
              <label className="text-sm text-zinc-600">Seed
                <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))}
                  className="ml-2 min-h-11 w-24 rounded-lg border border-zinc-300 px-2 font-mono text-sm" /></label>
            </div>
          )}
          {assets.some((a) => a.id !== leadId) && (
            <fieldset className="mt-4">
              <legend className="text-sm font-bold text-zinc-800">Also in the scene (optional)</legend>
              <div className="mt-2 flex flex-wrap gap-2">
                {assets.filter((a) => a.id !== leadId).map((a) => (
                  <button type="button" key={a.id} onClick={() => toggleExtra(a.id)} aria-pressed={extras.includes(a.id)}
                    className={`flex min-h-11 items-center gap-2 rounded-full border px-4 text-sm font-medium transition-colors ${extras.includes(a.id) ? "border-zinc-900 bg-zinc-900 text-white" : "border-zinc-300 bg-white hover:border-zinc-400 hover:bg-zinc-50"}`}>
                    <span className={`rounded-full px-1.5 py-0.5 font-mono text-[10px] ${KIND_BADGE[a.kind]}`}>{a.kind}</span>
                    {a.name}
                  </button>
                ))}
              </div>
            </fieldset>
          )}
          <button type="submit" disabled={!lead || busy}
            className="mt-4 min-h-11 rounded-full bg-zinc-900 px-6 text-sm font-semibold text-white transition-colors hover:bg-zinc-700 disabled:opacity-40">
            {busy ? "Generating (~30s)…" : "Generate"}
          </button>
        </section>
      </form>

      {graphic && (
        <section className="mt-6 rounded-xl border border-zinc-200 bg-white p-5 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-base font-semibold">{graphic.name} — {profile} graphic</h2>
            <span className="rounded-full bg-teal-800 px-2 py-0.5 font-mono text-[11px] text-white">
              {graphic.model} · {graphic.source === "modal-lora" ? "from the trained LoRA adapter" : graphic.source === "meta-image-edit" ? "from your reference image" : "from DNA description (no reference)"} · {graphic.output_hash.slice(0, 12)}…
            </span>
          </div>
          <img src={graphic.image} alt={`${graphic.name} generated`} className="mt-3 w-full max-w-xl rounded-lg border border-zinc-200 object-contain" />
          <div className="mt-3 flex gap-4 text-sm">
            <a href={graphic.image}
              download={`${graphic.name.replace(/\s+/g, "-")}.${graphic.image.includes("webp") ? "webp" : "png"}`}
              className="font-medium text-teal-800 underline">Download image</a>
            <button onClick={() => setGraphic(null)} className="text-zinc-600">Close</button>
          </div>
          {graphic.checks?.length > 0
            ? <p className="mt-2 text-[13px] text-amber-600">Format checks: {graphic.checks.join("; ")}</p>
            : graphic.profile === "tshirt" && <p className="mt-2 text-[13px] text-teal-700">Format checks passed: transparent background, subject contained.</p>}
          <p className="mt-2 font-mono text-[11px] leading-relaxed text-zinc-600">{graphic.prompt}</p>
        </section>
      )}

      <section className="mt-8">
        <h2 className="text-lg font-bold">Your outputs</h2>
        {projects.length === 0 && <p className="mt-2 text-sm text-zinc-600">No storyboard or video projects yet.</p>}
        <div className="mt-3 grid gap-4 sm:grid-cols-2">
          {projects.map((p) => (
            <article key={p.id} className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm">
              <div className="flex items-start justify-between gap-2">
                <Link href={`/projects/${p.id}`} className="text-[15px] font-semibold hover:underline">{p.title}</Link>
                <span className={`rounded-full px-2 py-0.5 font-mono text-[10px] font-bold ${p.status === "ready" ? "bg-teal-800 text-white" : "bg-zinc-200 text-zinc-700"}`}>{p.status}</span>
              </div>
              <p className="mt-1 line-clamp-2 text-sm leading-relaxed text-zinc-700">{p.premise || "No premise yet."}</p>
              <p className="mt-2 font-mono text-[11px] text-zinc-600">{p.profile || "storyboard"} · seed {p.seed} · {p.panel_count} panels</p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {p.asset_ids.map((id) => byId[id] && (
                  <span key={id} className="rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-700">{byId[id].name}</span>
                ))}
              </div>
              <div className="mt-3 flex gap-3 text-sm">
                <Link href={`/projects/${p.id}`} className="font-medium text-teal-800 underline">Open</Link>
                <button onClick={() => onDelete(p)} className="text-zinc-600 hover:text-red-700">Delete</button>
              </div>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
