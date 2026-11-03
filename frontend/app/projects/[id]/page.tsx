"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  Asset, Panel, ProjectDetail, exportProject, getProject, listAssets, runProject, updateProject,
} from "@/lib/api";

const AGENTS: Record<string, { name: string; role: string }> = {
  canon: { name: "Canon Keeper", role: "Freezes DNA into the hashed Bible" },
  story: { name: "Story Architect", role: "Premise → arc beats" },
  layout: { name: "Layout Director", role: "Camera, pose, expression, blocking" },
  render: { name: "Panel Renderer", role: "Draws panels strictly from the Bible" },
  critic: { name: "Consistency Critic", role: "Verifies panels against the Bible, reports drift" },
};

function scoreClass(s: number) {
  return s >= 95 ? "bg-teal-800 text-white" : s >= 80 ? "bg-amber-500 text-zinc-900" : "bg-red-700 text-white";
}

export default function ProjectPage({ params }: { params: { id: string } }) {
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [selected, setSelected] = useState<Panel | null>(null);
  const [seed, setSeed] = useState(42);
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const [lastHash, setLastHash] = useState("");
  const [verified, setVerified] = useState<string>("");


  const refresh = useCallback(async () => {
    try {
      const [p, a] = await Promise.all([getProject(params.id), listAssets()]);
      setProject(p); setAssets(a); setSeed(p.seed);
      setSelected((s) => p.panels.find((x) => x.id === s?.id) || p.panels[0] || null);
      const run = p.runs[p.runs.length - 1];
      if (run) setLastHash(run.output_hash);
      setError("");
    } catch { setError("Backend unreachable — is FastAPI running on :8000?"); }
  }, [params.id]);
  useEffect(() => { refresh(); }, [refresh]);

  async function onRun(verify = false) {
    if (!project || running) return;
    setRunning(true); setError(""); if (!verify) setVerified("");
    try {
      const run = await runProject(project.id, seed);
      if (verify) {
        setVerified(run.output_hash === lastHash
          ? `Deterministic ✓ — re-run reproduced output ${run.output_hash.slice(0, 10)} exactly.`
          : `Hash changed (${run.output_hash.slice(0, 10)}) — inputs or seed changed since the last run.`);
      }
      setLastHash(run.output_hash);
      await refresh();
    } catch (err) { setError(err instanceof Error ? err.message : "Pipeline run failed"); }
    finally { setRunning(false); }
  }

  async function toggleAsset(id: string) {
    if (!project) return;
    const ids = project.asset_ids.includes(id)
      ? project.asset_ids.filter((x) => x !== id) : [...project.asset_ids, id];
    await updateProject(project.id, { asset_ids: ids }); refresh();
  }

  async function onExport() {
    const data = await exportProject(params.id);
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url; a.download = `${project?.title.replace(/\s+/g, "-").toLowerCase() || "board"}.json`; a.click();
    URL.revokeObjectURL(url);
  }

  if (!project) return <main className="mx-auto max-w-6xl px-4 py-10 text-sm text-zinc-500">{error || "Loading story…"}</main>;
  const run = project.runs[project.runs.length - 1];
  const bibleAssets = run?.bible?.assets ? Object.values(run.bible.assets) : [];

  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-6">
      <Link href="/" className="text-sm font-medium text-teal-800 underline">← All stories</Link>
      <div className="mt-2 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold tracking-tight">{project.title}</h1>
          <p className="mt-1 max-w-2xl text-sm leading-relaxed text-zinc-600">{project.premise || "No premise yet."}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="text-sm text-zinc-600">Seed
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))}
              className="ml-2 min-h-11 w-24 rounded-md border border-zinc-300 bg-white px-2 font-mono text-sm" />
          </label>
          <span className="rounded-sm bg-zinc-200 px-2 py-1 font-mono text-[11px] text-zinc-700">{project.profile || "storyboard"}</span>
          <button onClick={() => onRun(false)} disabled={running}
            className="min-h-11 rounded-md bg-zinc-900 px-4 text-sm font-semibold text-white disabled:opacity-40">
            {running ? "Generating images (~30s/panel)…" : "Run pipeline"}
          </button>
          <button onClick={() => onRun(true)} disabled={running || !run}
            className="min-h-11 rounded-md border border-zinc-400 bg-white px-4 text-sm font-medium disabled:opacity-40">
            Verify determinism
          </button>
          <button onClick={onExport} disabled={!run}
            className="min-h-11 rounded-md border border-zinc-400 bg-white px-4 text-sm font-medium disabled:opacity-40">
            Export board JSON
          </button>
        </div>
      </div>

      {error && <div className="mt-4 rounded-md bg-red-50 px-3 py-2.5 text-sm text-red-800" role="alert">{error}</div>}
      {verified && <div className="mt-4 rounded-md bg-teal-50 px-3 py-2.5 font-mono text-[13px] text-teal-900" role="status">{verified}</div>}

      <section className="mt-5 border border-zinc-300 bg-white p-4">
        <h2 className="text-sm font-semibold">Cast &amp; world for this story</h2>
        <div className="mt-2 flex flex-wrap gap-2">
          {assets.map((a) => (
            <button key={a.id} onClick={() => toggleAsset(a.id)} aria-pressed={project.asset_ids.includes(a.id)}
              className={`min-h-11 rounded-md border px-3 text-sm ${project.asset_ids.includes(a.id) ? "border-zinc-900 bg-zinc-900 text-white" : "border-zinc-300 hover:bg-zinc-50"}`}>
              <span className="mr-1.5 font-mono text-[10px] opacity-70">{a.kind}</span>{a.name}
            </button>
          ))}
          {assets.length === 0 && <span className="text-sm text-zinc-500">No assets yet — <Link href="/assets" className="underline">define some</Link> or load the demo from Studio.</span>}
        </div>
        {run && (
          <p className="mt-3 font-mono text-[11px] leading-relaxed text-zinc-500">
            Bible {run.bible_hash.slice(0, 10)} · output {run.output_hash.slice(0, 10)} · seed {run.seed} ·
            consistency {run.consistency_avg}/100 · frozen assets: {bibleAssets.map((a) => `${a.name}@${a.dna_hash.slice(0, 6)}`).join(", ") || "—"}
          </p>
        )}
      </section>

      {run && (
        <section className="mt-5" aria-label="Pipeline">
          <h2 className="text-sm font-semibold">Pipeline — latest run</h2>
          <ol className="mt-2 grid gap-2 md:grid-cols-5">
            {[...run.stages].sort((a, b) => a.order - b.order).map((s) => (
              <li key={s.id} className="border border-zinc-300 bg-white p-3">
                <p className="font-mono text-[10px] text-zinc-500">AGENT {s.order + 1} · {s.status}</p>
                <p className="mt-0.5 text-sm font-semibold">{AGENTS[s.agent]?.name || s.agent}</p>
                <p className="text-[11px] text-zinc-500">{AGENTS[s.agent]?.role}</p>
                <p className="mt-1.5 text-[13px] leading-relaxed text-zinc-700">{s.summary}</p>
              </li>
            ))}
          </ol>
        </section>
      )}

      <section className="mt-6" aria-label="Board">
        <h2 className="text-sm font-semibold">
          Board {project.panels.length > 0 && <span className="font-normal text-zinc-500">— click a panel for its locked prompt and critic report</span>}
        </h2>
        {project.panels.length === 0 && (
          <p className="mt-2 border border-dashed border-zinc-400 bg-white px-4 py-8 text-center text-sm text-zinc-500">
            No panels yet. Pick at least one character above, then run the pipeline.
          </p>
        )}
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {project.panels.map((p) => (
            <button key={p.id} onClick={() => setSelected(p)}
              className={`border bg-white text-left ${selected?.id === p.id ? "border-zinc-900 outline outline-2 outline-zinc-900" : "border-zinc-300"}`}>
              {p.image
                ? <span className="block"><img src={p.image} alt={p.beat_title} className="block h-auto w-full object-cover" /></span>
                : <span className="block" dangerouslySetInnerHTML={{ __html: p.svg }} />}
              <span className="flex items-center justify-between px-2.5 py-2">
                <span className="text-[13px] font-semibold">{p.index + 1}. {p.beat_title}</span>
                <span className={`rounded-sm px-1.5 py-0.5 font-mono text-[10px] ${scoreClass(p.consistency_score)}`}>
                  {p.consistency_score}/100{p.repaired ? " · repaired" : ""}
                </span>
              </span>
              <span className="block px-2.5 pb-2 font-mono text-[10px] text-zinc-500">
                {p.camera.shot} · {p.camera.angle} · {p.camera.lens}
              </span>
            </button>
          ))}
        </div>
      </section>

      {selected && (
        <section className="mt-5 border border-zinc-300 bg-zinc-900 p-4 text-zinc-100" aria-label="Panel detail">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-sm font-semibold">Panel {selected.index + 1} — {selected.beat_title}</h2>
            <span className={`rounded-sm px-1.5 py-0.5 font-mono text-[10px] ${scoreClass(selected.consistency_score)}`}>
              consistency {selected.consistency_score}/100
            </span>
          </div>
          {selected.image && <img src={selected.image} alt={selected.beat_title} className="mt-3 w-full max-w-md border border-zinc-700 object-contain" />}
          <p className="mt-2 text-sm leading-relaxed">{selected.action}</p>
          {selected.dialogue && <p className="mt-1 text-sm italic text-zinc-300">{selected.dialogue}</p>}
          <p className="mt-3 font-mono text-[11px] leading-relaxed text-zinc-400">{selected.prompt}</p>
          {selected.issues.length > 0
            ? <p className="mt-2 text-[13px] text-amber-400">Critic issues: {selected.issues.join("; ")}</p>
            : <p className="mt-2 text-[13px] text-teal-400">Critic: every checked trait matches the frozen Bible.</p>}
        </section>
      )}
    </main>
  );
}
