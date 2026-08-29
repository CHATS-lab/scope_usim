-- Adds the variant and ended_reason columns to studysession without dropping
-- any pilot data. Idempotent: safe to re-run.
--
-- Apply on the VPS after deploying the new code:
--   docker compose -f deploy/docker-compose.yml exec -T postgres \
--     psql -U usim -d usim_study \
--     < deploy/migrations/001_variant_and_ended_reason.sql
--
-- SQLModel stores Enum columns as native Postgres ENUM types labeled by
-- the Python enum *member name* (e.g. V1, UNKNOWN). Matching that
-- convention here keeps the ORM happy.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'studyvariant') THEN
        CREATE TYPE studyvariant AS ENUM ('V1', 'V2');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'endedreason') THEN
        CREATE TYPE endedreason AS ENUM (
            'USER_STOP', 'AGENT_TRANSFERRED', 'TURN_LIMIT', 'UNKNOWN'
        );
    END IF;
END $$;

ALTER TABLE studysession
    ADD COLUMN IF NOT EXISTS variant studyvariant NOT NULL DEFAULT 'V1';

ALTER TABLE studysession
    ADD COLUMN IF NOT EXISTS ended_reason endedreason NOT NULL DEFAULT 'UNKNOWN';
