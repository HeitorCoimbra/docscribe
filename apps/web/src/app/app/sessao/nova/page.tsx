import { getServerSession } from "next-auth";
import { redirect } from "next/navigation";
import { authOptions } from "@/lib/auth";
import jwt from "jsonwebtoken";

export default async function NovaPage() {
  const session = await getServerSession(authOptions);
  if (!session) redirect("/auth/login");

  const apiUrl = process.env.API_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

  const token = jwt.sign(
    { email: session.user?.email, name: session.user?.name },
    process.env.NEXTAUTH_SECRET!,
    { algorithm: "HS256" }
  );

  const res = await fetch(`${apiUrl}/api/v1/threads`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({}),
  });

  if (!res.ok) redirect("/app");

  const thread = await res.json();
  redirect(`/app/sessao/${thread.id}`);
}
