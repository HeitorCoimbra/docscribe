import { getServerSession } from "next-auth";
import { redirect } from "next/navigation";
import { authOptions } from "@/lib/auth";
import jwt from "jsonwebtoken";
import { SessionView } from "@/components/session/SessionView";
import type { ThreadDetail } from "@/types/session";

export default async function SessionPage({ params }: { params: { id: string } }) {
  const session = await getServerSession(authOptions);
  if (!session) redirect("/auth/login");

  const apiUrl = process.env.API_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

  const token = jwt.sign(
    { email: session.user?.email, name: session.user?.name },
    process.env.NEXTAUTH_SECRET!,
    { algorithm: "HS256" }
  );

  const res = await fetch(`${apiUrl}/api/v1/threads/${params.id}`, {
    headers: { Authorization: `Bearer ${token}` },
    cache: 'no-store',
  });

  if (!res.ok) redirect("/app");

  const thread: ThreadDetail = await res.json();
  return <SessionView thread={thread} />;
}
