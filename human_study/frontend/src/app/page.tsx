import { Suspense } from "react";
import LandingClient from "./LandingClient";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <Suspense fallback={<main className="flex min-h-screen items-center justify-center" />}>
      <LandingClient />
    </Suspense>
  );
}
