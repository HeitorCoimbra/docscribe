'use client';
import { signIn } from "next-auth/react";
import { Button } from "@/components/ui/button";

export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-6 p-8">
        <div className="flex flex-col items-center gap-2">
          <h1 className="text-2xl font-semibold tracking-tight">DocScribe</h1>
          <p className="text-sm text-muted-foreground">
            Sumários de passagem de plantão para UTI
          </p>
        </div>
        <Button
          onClick={() => signIn("google", { callbackUrl: "/app" })}
          className="w-full"
        >
          Entrar com Google
        </Button>
      </div>
    </div>
  );
}
