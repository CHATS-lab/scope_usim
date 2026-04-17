import { Suspense } from "react";
import StudyClient from "./StudyClient";

export const dynamic = "force-dynamic";

export default function Page() {
  return (
    <Suspense fallback={<main className="flex min-h-screen items-center justify-center text-text">Loading...</main>}>
      <StudyClient />
    </Suspense>
  );
}
