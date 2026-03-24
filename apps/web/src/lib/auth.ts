import { NextAuthOptions } from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import jwt from "jsonwebtoken";

export const authOptions: NextAuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],
  session: {
    strategy: "jwt",
  },
  secret: process.env.NEXTAUTH_SECRET,
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.email = user.email;
        token.name = user.name;
        token.picture = user.image;
      }
      return token;
    },
    async session({ session, token }) {
      if (token) {
        if (session.user) session.user.email = token.email as string;
        // Sign a standard HS256 JWT that FastAPI/PyJWT can verify
        session.accessToken = jwt.sign(
          { email: token.email, name: token.name },
          process.env.NEXTAUTH_SECRET!,
          { algorithm: "HS256" }
        );
      }
      return session;
    },
  },
  pages: {
    signIn: "/auth/login",
  },
};

declare module "next-auth" {
  interface Session {
    accessToken?: string;
  }
}
