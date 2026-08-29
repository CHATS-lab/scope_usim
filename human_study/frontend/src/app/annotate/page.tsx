import { Suspense } from "react";
import AnnotateClient from "./AnnotateClient";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <Suspense
      fallback={
        <main className="flex min-h-screen items-center justify-center text-text">
          Loading…
        </main>
      }
    >
      <AnnotateClient />
    </Suspense>
  );
}
