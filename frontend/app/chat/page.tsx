"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Conversation, Message, createConversation, deleteConversation,
  getConversation, listConversations, renameConversation, streamMessage,
} from "@/lib/api";

export default function Home() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const refreshList = useCallback(async () => {
    try { setConversations(await listConversations()); } catch { setError("Backend unreachable — is FastAPI running on :8000?"); }
  }, []);

  const openConversation = useCallback(async (id: string) => {
    setActiveId(id); setDrawer(false); setError("");
    try { setMessages((await getConversation(id)).messages); } catch { setError("Could not load conversation."); }
  }, []);

  useEffect(() => { refreshList(); }, [refreshList]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, sending]);

  async function newChat() {
    const c = await createConversation();
    setConversations((p) => [c, ...p]); setActiveId(c.id); setMessages([]); setDrawer(false);
    inputRef.current?.focus();
  }

  async function send(e?: React.FormEvent) {
    e?.preventDefault();
    const text = input.trim();
    if (!text || sending) return;
    setError("");
    let convId = activeId;
    if (!convId) { const c = await createConversation(); convId = c.id; setActiveId(c.id); }
    setInput("");
    const optimistic: Message = { id: `tmp-${Date.now()}`, conversation_id: convId, role: "user", content: text, created_at: new Date().toISOString() };
    const placeholder: Message = { id: `tmp-a-${Date.now()}`, conversation_id: convId, role: "assistant", content: "", created_at: new Date().toISOString() };
    setMessages((p) => [...p, optimistic, placeholder]);
    setSending(true);
    try {
      await streamMessage(convId, text, (delta) => {
        setMessages((p) => p.map((m) => (m.id === placeholder.id ? { ...m, content: m.content + delta } : m)));
      }, () => {}, (msg) => {
        setMessages((p) => p.filter((m) => m.id !== placeholder.id));
        setError(msg);
      });
      const detail = await getConversation(convId);
      setMessages(detail.messages);
      await refreshList();
    } catch { setError((prev) => prev || "Message failed. Check the backend is running."); }
    finally { setSending(false); inputRef.current?.focus(); }
  }

  async function onRename(c: Conversation) {
    const title = window.prompt("Rename chat", c.title);
    if (title?.trim()) { await renameConversation(c.id, title.trim()); refreshList(); }
  }
  async function onDelete(c: Conversation) {
    if (!window.confirm(`Delete "${c.title}"?`)) return;
    await deleteConversation(c.id);
    if (activeId === c.id) { setActiveId(null); setMessages([]); }
    refreshList();
  }

  const sidebar = (
    <div className="flex h-full flex-col bg-zinc-900 text-zinc-100">
      <div className="flex items-center justify-between px-4 pt-5 pb-3">
        <span className="text-[15px] font-semibold tracking-tight">WREN Chat</span>
        <button onClick={() => setDrawer(false)} className="flex h-11 w-11 items-center justify-center rounded-lg text-xl md:hidden" aria-label="Close menu">×</button>
      </div>
      <div className="px-3"><button onClick={newChat} className="min-h-11 w-full rounded-lg bg-zinc-100 px-3 text-left text-sm font-medium text-zinc-900 active:bg-white">＋ New chat</button></div>
      <nav className="mt-3 flex-1 space-y-1 overflow-y-auto px-3 pb-4">
        {conversations.length === 0 && <p className="px-2 pt-4 text-sm text-zinc-400">No chats yet. Start your first one.</p>}
        {conversations.map((c) => (
          <div key={c.id} className={`group flex min-h-11 items-center gap-1 rounded-lg px-2 ${activeId === c.id ? "bg-zinc-800" : "hover:bg-zinc-800/60"}`}>
            <button onClick={() => openConversation(c.id)} className="min-w-0 flex-1 truncate py-2.5 text-left text-sm">{c.title}</button>
            <button onClick={() => onRename(c)} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-zinc-400 hover:text-white" aria-label="Rename">✎</button>
            <button onClick={() => onDelete(c)} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-zinc-400 hover:text-white" aria-label="Delete">🗑</button>
          </div>
        ))}
      </nav>
      <p className="px-4 pb-5 text-xs leading-relaxed text-zinc-500">Powered by Muse Spark.<br />Chats are saved in SQLite.</p>
    </div>
  );

  return (
    <div className="flex h-dvh w-full overflow-hidden">
      {/* Desktop sidebar */}
      <aside className="hidden w-72 shrink-0 md:block">{sidebar}</aside>
      {/* Mobile drawer */}
      {drawer && (
        <div className="fixed inset-0 z-40 md:hidden">
          <div className="absolute inset-0 bg-black/50" onClick={() => setDrawer(false)} />
          <aside className="absolute inset-y-0 left-0 w-[85vw] max-w-80 shadow-xl">{sidebar}</aside>
        </div>
      )}

      <main className="flex min-w-0 flex-1 flex-col bg-white">
        <header className="flex min-h-14 items-center gap-2 border-b border-zinc-200 px-3 pt-[env(safe-area-inset-top)]">
          <button onClick={() => setDrawer(true)} className="flex h-11 w-11 items-center justify-center rounded-lg text-xl md:hidden" aria-label="Open menu">☰</button>
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-[15px] font-semibold">{conversations.find((c) => c.id === activeId)?.title || "New chat"}</h1>
            <p className="text-xs text-zinc-500">Muse Spark · {sending ? "Replying…" : "SQLite history on"}</p>
          </div>
          <button onClick={newChat} className="min-h-11 rounded-lg px-3 text-sm font-medium text-zinc-700 hover:bg-zinc-100 md:hidden">＋ New</button>
        </header>

        {error && <div className="mx-3 mt-3 rounded-lg bg-red-50 px-3 py-2.5 text-sm text-red-800" role="alert">{error}</div>}

        <div className="flex-1 overflow-y-auto px-3 py-4 sm:px-5">
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-3">
            {messages.length === 0 && (
              <div className="pt-16 text-center">
                <h2 className="text-2xl font-semibold tracking-tight">What can I help with?</h2>
                <p className="mx-auto mt-2 max-w-sm text-sm leading-relaxed text-zinc-500">Ask anything. Your conversation is saved automatically and will be here when you come back.</p>
                <div className="mx-auto mt-6 grid max-w-md gap-2 text-left">
                  {["Explain Llama models in simple terms", "Write a hackathon pitch in 3 lines", "Give me 5 demo ideas for a WREN app"].map((s) => (
                    <button key={s} onClick={() => { setInput(s); inputRef.current?.focus(); }} className="min-h-11 rounded-lg border border-zinc-200 px-3 py-2.5 text-sm active:bg-zinc-50">{s}</button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m) => (
              <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-[88%] whitespace-pre-wrap break-words px-3.5 py-2.5 text-[15px] leading-relaxed sm:max-w-[75%] ${m.role === "user" ? "rounded-2xl rounded-br-md bg-zinc-900 text-white" : "rounded-2xl rounded-bl-md bg-zinc-100 text-zinc-900"}`}>
                  {m.content || (sending && m.role === "assistant" ? "…" : "")}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        <form onSubmit={send} className="border-t border-zinc-200 bg-white px-3 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
          <div className="mx-auto flex w-full max-w-3xl items-end gap-2">
            <textarea ref={inputRef} value={input} onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              rows={1} placeholder="Message WREN Chat…" aria-label="Message"
              className="max-h-36 min-h-11 flex-1 resize-none rounded-xl border border-zinc-300 px-3.5 py-2.5 text-[16px] outline-none focus:border-zinc-500" />
            <button type="submit" disabled={!input.trim() || sending} className="flex min-h-11 min-w-11 items-center justify-center rounded-xl bg-zinc-900 px-4 text-sm font-medium text-white disabled:opacity-40" aria-label="Send">↑</button>
          </div>
          <p className="mx-auto mt-2 max-w-3xl text-center text-xs text-zinc-400">Enter to send · Shift+Enter for a new line</p>
        </form>
      </main>
    </div>
  );
}
