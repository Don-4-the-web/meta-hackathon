"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { login, register, setToken } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState<"login" | "register">("login");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const fn = mode === "login" ? login : register;
      const { token } = await fn(username.trim(), password);
      setToken(token);
      router.push("/assets");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-[70vh] w-full max-w-md flex-col justify-center px-4">
      <h1 className="text-xl font-bold">
        {mode === "login" ? "Log in" : "Create an account"}
      </h1>
      <p className="mt-1 text-sm text-zinc-600">
        Each account trains its own LoRAs — your mascots stay private to you.
      </p>
      <form onSubmit={submit} className="mt-5 space-y-3 rounded-xl border bg-white p-5 shadow-sm">
        <label className="block text-sm font-medium">
          Username
          <input
            value={username} onChange={(e) => setUsername(e.target.value)}
            autoComplete="username" required minLength={3}
            className="mt-1 w-full rounded-md border px-3 py-2 text-sm"
          />
        </label>
        <label className="block text-sm font-medium">
          Password
          <input
            type="password" value={password} onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            required minLength={6}
            className="mt-1 w-full rounded-md border px-3 py-2 text-sm"
          />
        </label>
        {error && (
          <p role="alert" className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>
        )}
        <button
          type="submit" disabled={busy || !username.trim() || !password}
          className="w-full rounded-md bg-zinc-900 px-3 py-2 text-sm font-semibold text-white disabled:opacity-50"
        >
          {busy ? "Working…" : mode === "login" ? "Log in" : "Sign up"}
        </button>
        <button
          type="button" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(null); }}
          className="w-full text-center text-sm text-zinc-600 underline"
        >
          {mode === "login" ? "Need an account? Sign up" : "Already have an account? Log in"}
        </button>
      </form>
    </main>
  );
}
