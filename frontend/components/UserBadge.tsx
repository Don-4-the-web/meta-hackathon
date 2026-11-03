"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { getMe, getToken, logout, type User } from "@/lib/api";

export default function UserBadge() {
  const [user, setUser] = useState<User | null>(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (!getToken()) { setChecked(true); return; }
    getMe().then(setUser).catch(() => setUser(null)).finally(() => setChecked(true));
  }, []);

  if (!checked) return null;
  if (!user) {
    return (
      <Link href="/login"
        className="ml-auto rounded-md bg-amber-400 px-3 py-1.5 text-sm font-semibold text-zinc-900 hover:bg-amber-300">
        Log in
      </Link>
    );
  }
  return (
    <div className="ml-auto flex items-center gap-2 text-sm">
      <span className="hidden font-medium text-zinc-200 sm:block">{user.username}</span>
      <span className="rounded-full bg-zinc-800 px-2 py-0.5 font-mono text-[11px] text-zinc-300" title="LoRA trainings used">
        {user.trainings_used}/{user.max_trainings} trains
      </span>
      <button
        onClick={() => { void logout().then(() => { setUser(null); window.location.href = "/login"; }); }}
        className="rounded-md px-2 py-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-white"
      >
        Log out
      </button>
    </div>
  );
}
