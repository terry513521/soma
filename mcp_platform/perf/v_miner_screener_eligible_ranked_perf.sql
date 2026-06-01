-- Benchmark queries for replacing v_miner_screener_eligible_ranked.
--
-- Usage in pgAdmin:
--   Copy-paste and run all statements in Query Tool.
--
-- Notes:
-- - Baseline section matches current mcp_platform usage.
-- - Candidate 1 is the simplest replacement (v_miner_status only).
-- - Candidate 2 is closer to current ranking semantics while still avoiding
--   v_miner_screener_eligible_ranked.

-- =====================================================
-- BASELINE (current behavior)
-- =====================================================

-- B1) Count eligible rows (used to compute top_limit)
SELECT COUNT(*) AS total_eligible
FROM v_miner_screener_eligible_ranked r
WHERE r.competition_id = 40;

-- B2) Top miner IDs for _select_miner_ss58 (utils.py)
WITH params AS (
    SELECT COALESCE(CEIL(COUNT(*) * 0.2::numeric)::int, 0) AS top_limit
    FROM v_miner_screener_eligible_ranked r
    WHERE r.competition_id = 40
)
SELECT r.miner_id
FROM v_miner_screener_eligible_ranked r
CROSS JOIN params p
WHERE r.competition_id = 40
  AND r.rank <= p.top_limit
ORDER BY r.rank ASC;

-- B3) Top ss58 for _load_top_screener_uids_for_competition (validator.py)
WITH params AS (
    SELECT COALESCE(CEIL(COUNT(*) * 0.2::numeric)::int, 0) AS top_limit
    FROM v_miner_screener_eligible_ranked r
    WHERE r.competition_id = 40
)
SELECT m.ss58
FROM v_miner_screener_eligible_ranked r
JOIN miners m ON m.id = r.miner_id
CROSS JOIN params p
WHERE r.competition_id = 40
  AND r.rank <= p.top_limit
  AND m.miner_banned_status IS FALSE
ORDER BY r.rank ASC;


-- =====================================================
-- CANDIDATE 1 (simplest): v_miner_status only (live view)
-- =====================================================
-- Potential behavior difference:
-- - Uses v_miner_status.screener_rank tie-breaking rules instead of
--   v_miner_screener_eligible_ranked rank ordering.

-- C1.1) Count eligible from v_miner_status
SELECT COUNT(*) AS total_eligible
FROM v_miner_status s
WHERE s.competition_id = 40
  AND s.is_banned IS FALSE
  AND s.screener_rank IS NOT NULL;

-- C1.2) Top miner IDs for _select_miner_ss58 replacement
WITH params AS (
    SELECT COALESCE(CEIL(COUNT(*) * 0.2::numeric)::int, 0) AS top_limit
    FROM v_miner_status s
    WHERE s.competition_id = 40
      AND s.is_banned IS FALSE
      AND s.screener_rank IS NOT NULL
)
SELECT m.id AS miner_id
FROM v_miner_status s
JOIN miners m ON m.ss58 = s.ss58
CROSS JOIN params p
WHERE s.competition_id = 40
  AND s.is_banned IS FALSE
  AND s.screener_rank IS NOT NULL
  AND s.screener_rank <= p.top_limit
ORDER BY s.screener_rank ASC;

-- C1.3) Top ss58 for _load_top_screener_uids_for_competition replacement
WITH params AS (
    SELECT COALESCE(CEIL(COUNT(*) * 0.2::numeric)::int, 0) AS top_limit
    FROM v_miner_status s
    WHERE s.competition_id = 40
      AND s.is_banned IS FALSE
      AND s.screener_rank IS NOT NULL
)
SELECT s.ss58
FROM v_miner_status s
CROSS JOIN params p
WHERE s.competition_id = 40
  AND s.is_banned IS FALSE
  AND s.screener_rank IS NOT NULL
  AND s.screener_rank <= p.top_limit
ORDER BY s.screener_rank ASC;


-- =====================================================
-- CANDIDATE 2 (closer semantics, live only):
-- base tables + v_miner_status, re-rank eligible subset
-- =====================================================
-- This keeps ordering by:
--   total_screener_score DESC, first_upload_at ASC, ss58 ASC
-- matching v_miner_screener_eligible_ranked ordering.

-- C2.1) Count eligible rows
WITH screener_stats AS (
    SELECT
        mu.competition_fk AS competition_id,
        m.id AS miner_id,
        m.ss58,
        (
            SUM(
                CASE
                    WHEN bcs.id IS NOT NULL
                    THEN (bcs.score::float / SQRT(bc.compression_ratio::float))
                    ELSE 0.0
                END
            )
            /
            NULLIF(
                SUM(
                    CASE
                        WHEN bcs.id IS NOT NULL
                        THEN (1.0 / SQRT(bc.compression_ratio::float))
                        ELSE 0.0
                    END
                ),
                0
            )
        ) AS total_screener_score,
        COUNT(DISTINCT bcs.batch_challenge_fk) AS screener_scored,
        MIN(mu.created_at) AS first_upload_at
    FROM challenge_batches cb
    JOIN scripts s
      ON s.id = cb.script_fk
    JOIN miners m
      ON m.id = s.miner_fk
    JOIN miner_uploads mu
      ON mu.script_fk = s.id
    JOIN batch_challenges bc
      ON bc.challenge_batch_fk = cb.id
    JOIN screeners scr
      ON scr.competition_fk = mu.competition_fk
     AND scr.is_active IS TRUE
    JOIN screening_challenges sc
      ON sc.screener_fk = scr.id
     AND sc.challenge_fk = bc.challenge_fk
    LEFT JOIN batch_challenge_scores bcs
      ON bcs.batch_challenge_fk = bc.id
    WHERE mu.competition_fk = 40
    GROUP BY mu.competition_fk, m.id, m.ss58
),
eligible AS (
    SELECT ss.ss58
    FROM screener_stats ss
    JOIN v_miner_status ms
      ON ms.competition_id = ss.competition_id
     AND ms.ss58 = ss.ss58
    WHERE ms.is_banned IS FALSE
      AND COALESCE(ms.screener_challenges, 0) > 0
      AND COALESCE(ms.scored_screened_challenges, 0) >= COALESCE(ms.screener_challenges, 0)
)
SELECT COUNT(*) AS total_eligible
FROM eligible;

-- C2.2) Top miner IDs for _select_miner_ss58 replacement
WITH screener_stats AS (
    SELECT
        mu.competition_fk AS competition_id,
        m.id AS miner_id,
        m.ss58,
        (
            SUM(
                CASE
                    WHEN bcs.id IS NOT NULL
                    THEN (bcs.score::float / SQRT(bc.compression_ratio::float))
                    ELSE 0.0
                END
            )
            /
            NULLIF(
                SUM(
                    CASE
                        WHEN bcs.id IS NOT NULL
                        THEN (1.0 / SQRT(bc.compression_ratio::float))
                        ELSE 0.0
                    END
                ),
                0
            )
        ) AS total_screener_score,
        MIN(mu.created_at) AS first_upload_at
    FROM challenge_batches cb
    JOIN scripts s
      ON s.id = cb.script_fk
    JOIN miners m
      ON m.id = s.miner_fk
    JOIN miner_uploads mu
      ON mu.script_fk = s.id
    JOIN batch_challenges bc
      ON bc.challenge_batch_fk = cb.id
    JOIN screeners scr
      ON scr.competition_fk = mu.competition_fk
     AND scr.is_active IS TRUE
    JOIN screening_challenges sc
      ON sc.screener_fk = scr.id
     AND sc.challenge_fk = bc.challenge_fk
    LEFT JOIN batch_challenge_scores bcs
      ON bcs.batch_challenge_fk = bc.id
    WHERE mu.competition_fk = 40
    GROUP BY mu.competition_fk, m.id, m.ss58
),
eligible AS (
    SELECT
        ss.competition_id,
        ss.ss58,
        ss.miner_id,
        ss.total_screener_score,
        ss.first_upload_at
    FROM screener_stats ss
    JOIN v_miner_status ms
      ON ms.competition_id = ss.competition_id
     AND ms.ss58 = ss.ss58
    WHERE ms.is_banned IS FALSE
      AND COALESCE(ms.screener_challenges, 0) > 0
      AND COALESCE(ms.scored_screened_challenges, 0) >= COALESCE(ms.screener_challenges, 0)
),
ranked AS (
    SELECT
        e.competition_id,
        e.ss58,
        e.miner_id,
        ROW_NUMBER() OVER (
            PARTITION BY e.competition_id
            ORDER BY e.total_screener_score DESC NULLS LAST,
                     e.first_upload_at ASC NULLS FIRST,
                     e.ss58 ASC
        ) AS rank,
        COUNT(*) OVER (PARTITION BY e.competition_id) AS total_eligible
    FROM eligible e
),
params AS (
    SELECT COALESCE(CEIL(MAX(total_eligible) * 0.2::numeric)::int, 0) AS top_limit
    FROM ranked
)
SELECT r.miner_id
FROM ranked r
CROSS JOIN params p
WHERE r.rank <= p.top_limit
ORDER BY r.rank ASC;

-- C2.3) Top ss58 for _load_top_screener_uids_for_competition replacement
WITH screener_stats AS (
    SELECT
        mu.competition_fk AS competition_id,
        m.id AS miner_id,
        m.ss58,
        (
            SUM(
                CASE
                    WHEN bcs.id IS NOT NULL
                    THEN (bcs.score::float / SQRT(bc.compression_ratio::float))
                    ELSE 0.0
                END
            )
            /
            NULLIF(
                SUM(
                    CASE
                        WHEN bcs.id IS NOT NULL
                        THEN (1.0 / SQRT(bc.compression_ratio::float))
                        ELSE 0.0
                    END
                ),
                0
            )
        ) AS total_screener_score,
        MIN(mu.created_at) AS first_upload_at
    FROM challenge_batches cb
    JOIN scripts s
      ON s.id = cb.script_fk
    JOIN miners m
      ON m.id = s.miner_fk
    JOIN miner_uploads mu
      ON mu.script_fk = s.id
    JOIN batch_challenges bc
      ON bc.challenge_batch_fk = cb.id
    JOIN screeners scr
      ON scr.competition_fk = mu.competition_fk
     AND scr.is_active IS TRUE
    JOIN screening_challenges sc
      ON sc.screener_fk = scr.id
     AND sc.challenge_fk = bc.challenge_fk
    LEFT JOIN batch_challenge_scores bcs
      ON bcs.batch_challenge_fk = bc.id
    WHERE mu.competition_fk = 40
    GROUP BY mu.competition_fk, m.id, m.ss58
),
eligible AS (
    SELECT
        ss.competition_id,
        ss.ss58,
        ss.total_screener_score,
        ss.first_upload_at
    FROM screener_stats ss
    JOIN v_miner_status ms
      ON ms.competition_id = ss.competition_id
     AND ms.ss58 = ss.ss58
    WHERE ms.is_banned IS FALSE
      AND COALESCE(ms.screener_challenges, 0) > 0
      AND COALESCE(ms.scored_screened_challenges, 0) >= COALESCE(ms.screener_challenges, 0)
),
ranked AS (
    SELECT
        e.competition_id,
        e.ss58,
        ROW_NUMBER() OVER (
            PARTITION BY e.competition_id
            ORDER BY e.total_screener_score DESC NULLS LAST,
                     e.first_upload_at ASC NULLS FIRST,
                     e.ss58 ASC
        ) AS rank,
        COUNT(*) OVER (PARTITION BY e.competition_id) AS total_eligible
    FROM eligible e
),
params AS (
    SELECT COALESCE(CEIL(MAX(total_eligible) * 0.2::numeric)::int, 0) AS top_limit
    FROM ranked
)
SELECT r.ss58
FROM ranked r
CROSS JOIN params p
WHERE r.rank <= p.top_limit
ORDER BY r.rank ASC;

-- All candidate queries above use live views/tables only (no materialized views).
