-- No-DDL replacement testing for v_miner_screener_eligible_ranked.
-- pgAdmin-ready: SELECT-only (no CREATE/DROP).
-- Uses smaller logical blocks as CTEs:
--   v_screener_required_pairs_live
--   v_screener_first_script_live
--   v_screener_stats_live
--   v_screener_eligible_base_live
--   v_screener_eligible_ranked_live
-- Fixed params: competition_id = 40, top_fraction = 0.2

-- =====================================================
-- Q1) Equivalent to:
-- select(V_MINER_SCREENER_ELIGIBLE_RANKED.c.total_eligible)
-- .where(competition_id == 40).limit(1)
-- =====================================================
WITH
v_screener_required_pairs_live AS (
    WITH screener_counts AS (
        SELECT
            s.competition_fk AS competition_id,
            COUNT(DISTINCT sc.challenge_fk) AS screener_challenge_count
        FROM screeners s
        JOIN screening_challenges sc
          ON sc.screener_fk = s.id
        WHERE s.is_active IS TRUE
        GROUP BY s.competition_fk
    ),
    ratio_counts AS (
        SELECT
            cc.competition_fk AS competition_id,
            COALESCE(json_array_length(ccc.compression_ratios), 1) AS ratio_count
        FROM competition_configs cc
        LEFT JOIN compression_competition_config ccc
          ON ccc.competition_config_fk = cc.id
    )
    SELECT
        sc.competition_id,
        (sc.screener_challenge_count * rc.ratio_count)::bigint AS screener_required
    FROM screener_counts sc
    JOIN ratio_counts rc
      ON rc.competition_id = sc.competition_id
),
v_screener_first_script_live AS (
    WITH ranked_uploads AS (
        SELECT
            mu.competition_fk AS competition_id,
            s.miner_fk AS miner_id,
            mu.script_fk AS script_id,
            ROW_NUMBER() OVER (
                PARTITION BY mu.competition_fk, s.miner_fk
                ORDER BY mu.created_at ASC, mu.script_fk ASC
            ) AS rn
        FROM miner_uploads mu
        JOIN scripts s
          ON s.id = mu.script_fk
        WHERE mu.competition_fk IS NOT NULL
    )
    SELECT competition_id, miner_id, script_id
    FROM ranked_uploads
    WHERE rn = 1
),
v_screener_stats_live AS (
    SELECT
        mu.competition_fk AS competition_id,
        m.id AS miner_id,
        m.ss58,
        m.miner_banned_status AS is_banned,
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
        ) AS avg_score,
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
    JOIN screening_challenges sc
      ON sc.screener_fk = scr.id
     AND sc.challenge_fk = bc.challenge_fk
    LEFT JOIN batch_challenge_scores bcs
      ON bcs.batch_challenge_fk = bc.id
    WHERE scr.is_active IS TRUE
    GROUP BY mu.competition_fk, m.id, m.ss58, m.miner_banned_status
),
v_screener_eligible_base_live AS (
    SELECT
        ss.competition_id,
        ss.ss58,
        fs.miner_id,
        fs.script_id,
        ss.avg_score,
        ss.first_upload_at,
        ss.screener_scored,
        rp.screener_required
    FROM v_screener_stats_live ss
    JOIN v_screener_required_pairs_live rp
      ON rp.competition_id = ss.competition_id
    JOIN v_screener_first_script_live fs
      ON fs.competition_id = ss.competition_id
     AND fs.miner_id = ss.miner_id
    JOIN miners m
      ON m.ss58 = ss.ss58
    WHERE rp.screener_required > 0
      AND ss.screener_scored >= rp.screener_required
      AND m.miner_banned_status IS FALSE
),
v_screener_eligible_ranked_live AS (
    SELECT
        e.competition_id,
        e.ss58,
        e.miner_id,
        e.script_id,
        e.avg_score,
        e.first_upload_at,
        e.screener_scored,
        e.screener_required,
        ROW_NUMBER() OVER (
            PARTITION BY e.competition_id
            ORDER BY e.avg_score DESC NULLS LAST,
                     e.first_upload_at ASC NULLS FIRST,
                     e.ss58 ASC
        ) AS rank,
        COUNT(*) OVER (PARTITION BY e.competition_id) AS total_eligible
    FROM v_screener_eligible_base_live e
)
SELECT r.total_eligible
FROM v_screener_eligible_ranked_live r
WHERE r.competition_id = 40
LIMIT 1;


-- =====================================================
-- Q2) Total eligible (count-based)
-- =====================================================
WITH
v_screener_required_pairs_live AS (
    WITH screener_counts AS (
        SELECT s.competition_fk AS competition_id, COUNT(DISTINCT sc.challenge_fk) AS screener_challenge_count
        FROM screeners s
        JOIN screening_challenges sc ON sc.screener_fk = s.id
        WHERE s.is_active IS TRUE
        GROUP BY s.competition_fk
    ),
    ratio_counts AS (
        SELECT cc.competition_fk AS competition_id, COALESCE(json_array_length(ccc.compression_ratios), 1) AS ratio_count
        FROM competition_configs cc
        LEFT JOIN compression_competition_config ccc ON ccc.competition_config_fk = cc.id
    )
    SELECT sc.competition_id, (sc.screener_challenge_count * rc.ratio_count)::bigint AS screener_required
    FROM screener_counts sc
    JOIN ratio_counts rc ON rc.competition_id = sc.competition_id
),
v_screener_first_script_live AS (
    WITH ranked_uploads AS (
        SELECT
            mu.competition_fk AS competition_id,
            s.miner_fk AS miner_id,
            mu.script_fk AS script_id,
            ROW_NUMBER() OVER (PARTITION BY mu.competition_fk, s.miner_fk ORDER BY mu.created_at ASC, mu.script_fk ASC) AS rn
        FROM miner_uploads mu
        JOIN scripts s ON s.id = mu.script_fk
        WHERE mu.competition_fk IS NOT NULL
    )
    SELECT competition_id, miner_id, script_id
    FROM ranked_uploads
    WHERE rn = 1
),
v_screener_stats_live AS (
    SELECT
        mu.competition_fk AS competition_id,
        m.id AS miner_id,
        m.ss58,
        m.miner_banned_status AS is_banned,
        (
            SUM(CASE WHEN bcs.id IS NOT NULL THEN (bcs.score::float / SQRT(bc.compression_ratio::float)) ELSE 0.0 END)
            /
            NULLIF(SUM(CASE WHEN bcs.id IS NOT NULL THEN (1.0 / SQRT(bc.compression_ratio::float)) ELSE 0.0 END), 0)
        ) AS avg_score,
        COUNT(DISTINCT bcs.batch_challenge_fk) AS screener_scored,
        MIN(mu.created_at) AS first_upload_at
    FROM challenge_batches cb
    JOIN scripts s ON s.id = cb.script_fk
    JOIN miners m ON m.id = s.miner_fk
    JOIN miner_uploads mu ON mu.script_fk = s.id
    JOIN batch_challenges bc ON bc.challenge_batch_fk = cb.id
    JOIN screeners scr ON scr.competition_fk = mu.competition_fk
    JOIN screening_challenges sc ON sc.screener_fk = scr.id AND sc.challenge_fk = bc.challenge_fk
    LEFT JOIN batch_challenge_scores bcs ON bcs.batch_challenge_fk = bc.id
    WHERE scr.is_active IS TRUE
    GROUP BY mu.competition_fk, m.id, m.ss58, m.miner_banned_status
),
v_screener_eligible_base_live AS (
    SELECT
        ss.competition_id,
        ss.ss58,
        fs.miner_id,
        fs.script_id,
        ss.avg_score,
        ss.first_upload_at,
        ss.screener_scored,
        rp.screener_required
    FROM v_screener_stats_live ss
    JOIN v_screener_required_pairs_live rp ON rp.competition_id = ss.competition_id
    JOIN v_screener_first_script_live fs ON fs.competition_id = ss.competition_id AND fs.miner_id = ss.miner_id
    JOIN miners m ON m.ss58 = ss.ss58
    WHERE rp.screener_required > 0
      AND ss.screener_scored >= rp.screener_required
      AND m.miner_banned_status IS FALSE
)
SELECT COUNT(*) AS total_eligible
FROM v_screener_eligible_base_live r
WHERE r.competition_id = 40;


-- =====================================================
-- Q3) Top miner_ids (competition_id = 40, top_fraction = 0.2)
-- =====================================================
WITH
v_screener_required_pairs_live AS (
    WITH screener_counts AS (
        SELECT s.competition_fk AS competition_id, COUNT(DISTINCT sc.challenge_fk) AS screener_challenge_count
        FROM screeners s
        JOIN screening_challenges sc ON sc.screener_fk = s.id
        WHERE s.is_active IS TRUE
        GROUP BY s.competition_fk
    ),
    ratio_counts AS (
        SELECT cc.competition_fk AS competition_id, COALESCE(json_array_length(ccc.compression_ratios), 1) AS ratio_count
        FROM competition_configs cc
        LEFT JOIN compression_competition_config ccc ON ccc.competition_config_fk = cc.id
    )
    SELECT sc.competition_id, (sc.screener_challenge_count * rc.ratio_count)::bigint AS screener_required
    FROM screener_counts sc
    JOIN ratio_counts rc ON rc.competition_id = sc.competition_id
),
v_screener_first_script_live AS (
    WITH ranked_uploads AS (
        SELECT
            mu.competition_fk AS competition_id,
            s.miner_fk AS miner_id,
            mu.script_fk AS script_id,
            ROW_NUMBER() OVER (PARTITION BY mu.competition_fk, s.miner_fk ORDER BY mu.created_at ASC, mu.script_fk ASC) AS rn
        FROM miner_uploads mu
        JOIN scripts s ON s.id = mu.script_fk
        WHERE mu.competition_fk IS NOT NULL
    )
    SELECT competition_id, miner_id, script_id
    FROM ranked_uploads
    WHERE rn = 1
),
v_screener_stats_live AS (
    SELECT
        mu.competition_fk AS competition_id,
        m.id AS miner_id,
        m.ss58,
        m.miner_banned_status AS is_banned,
        (
            SUM(CASE WHEN bcs.id IS NOT NULL THEN (bcs.score::float / SQRT(bc.compression_ratio::float)) ELSE 0.0 END)
            /
            NULLIF(SUM(CASE WHEN bcs.id IS NOT NULL THEN (1.0 / SQRT(bc.compression_ratio::float)) ELSE 0.0 END), 0)
        ) AS avg_score,
        COUNT(DISTINCT bcs.batch_challenge_fk) AS screener_scored,
        MIN(mu.created_at) AS first_upload_at
    FROM challenge_batches cb
    JOIN scripts s ON s.id = cb.script_fk
    JOIN miners m ON m.id = s.miner_fk
    JOIN miner_uploads mu ON mu.script_fk = s.id
    JOIN batch_challenges bc ON bc.challenge_batch_fk = cb.id
    JOIN screeners scr ON scr.competition_fk = mu.competition_fk
    JOIN screening_challenges sc ON sc.screener_fk = scr.id AND sc.challenge_fk = bc.challenge_fk
    LEFT JOIN batch_challenge_scores bcs ON bcs.batch_challenge_fk = bc.id
    WHERE scr.is_active IS TRUE
    GROUP BY mu.competition_fk, m.id, m.ss58, m.miner_banned_status
),
v_screener_eligible_base_live AS (
    SELECT
        ss.competition_id,
        ss.ss58,
        fs.miner_id,
        fs.script_id,
        ss.avg_score,
        ss.first_upload_at,
        ss.screener_scored,
        rp.screener_required
    FROM v_screener_stats_live ss
    JOIN v_screener_required_pairs_live rp ON rp.competition_id = ss.competition_id
    JOIN v_screener_first_script_live fs ON fs.competition_id = ss.competition_id AND fs.miner_id = ss.miner_id
    JOIN miners m ON m.ss58 = ss.ss58
    WHERE rp.screener_required > 0
      AND ss.screener_scored >= rp.screener_required
      AND m.miner_banned_status IS FALSE
),
v_screener_eligible_ranked_live AS (
    SELECT
        e.competition_id,
        e.ss58,
        e.miner_id,
        e.script_id,
        e.avg_score,
        e.first_upload_at,
        e.screener_scored,
        e.screener_required,
        ROW_NUMBER() OVER (
            PARTITION BY e.competition_id
            ORDER BY e.avg_score DESC NULLS LAST,
                     e.first_upload_at ASC NULLS FIRST,
                     e.ss58 ASC
        ) AS rank,
        COUNT(*) OVER (PARTITION BY e.competition_id) AS total_eligible
    FROM v_screener_eligible_base_live e
),
params AS (
    SELECT COALESCE(CEIL(COUNT(*) * 0.2::numeric)::int, 0) AS top_limit
    FROM v_screener_eligible_base_live r
    WHERE r.competition_id = 40
)
SELECT r.miner_id
FROM v_screener_eligible_ranked_live r
CROSS JOIN params p
WHERE r.competition_id = 40
  AND r.rank <= p.top_limit
ORDER BY r.rank ASC;


-- =====================================================
-- Q4) Top ss58 (competition_id = 40, top_fraction = 0.2)
-- =====================================================
WITH
v_screener_required_pairs_live AS (
    WITH screener_counts AS (
        SELECT s.competition_fk AS competition_id, COUNT(DISTINCT sc.challenge_fk) AS screener_challenge_count
        FROM screeners s
        JOIN screening_challenges sc ON sc.screener_fk = s.id
        WHERE s.is_active IS TRUE
        GROUP BY s.competition_fk
    ),
    ratio_counts AS (
        SELECT cc.competition_fk AS competition_id, COALESCE(json_array_length(ccc.compression_ratios), 1) AS ratio_count
        FROM competition_configs cc
        LEFT JOIN compression_competition_config ccc ON ccc.competition_config_fk = cc.id
    )
    SELECT sc.competition_id, (sc.screener_challenge_count * rc.ratio_count)::bigint AS screener_required
    FROM screener_counts sc
    JOIN ratio_counts rc ON rc.competition_id = sc.competition_id
),
v_screener_first_script_live AS (
    WITH ranked_uploads AS (
        SELECT
            mu.competition_fk AS competition_id,
            s.miner_fk AS miner_id,
            mu.script_fk AS script_id,
            ROW_NUMBER() OVER (PARTITION BY mu.competition_fk, s.miner_fk ORDER BY mu.created_at ASC, mu.script_fk ASC) AS rn
        FROM miner_uploads mu
        JOIN scripts s ON s.id = mu.script_fk
        WHERE mu.competition_fk IS NOT NULL
    )
    SELECT competition_id, miner_id, script_id
    FROM ranked_uploads
    WHERE rn = 1
),
v_screener_stats_live AS (
    SELECT
        mu.competition_fk AS competition_id,
        m.id AS miner_id,
        m.ss58,
        m.miner_banned_status AS is_banned,
        (
            SUM(CASE WHEN bcs.id IS NOT NULL THEN (bcs.score::float / SQRT(bc.compression_ratio::float)) ELSE 0.0 END)
            /
            NULLIF(SUM(CASE WHEN bcs.id IS NOT NULL THEN (1.0 / SQRT(bc.compression_ratio::float)) ELSE 0.0 END), 0)
        ) AS avg_score,
        COUNT(DISTINCT bcs.batch_challenge_fk) AS screener_scored,
        MIN(mu.created_at) AS first_upload_at
    FROM challenge_batches cb
    JOIN scripts s ON s.id = cb.script_fk
    JOIN miners m ON m.id = s.miner_fk
    JOIN miner_uploads mu ON mu.script_fk = s.id
    JOIN batch_challenges bc ON bc.challenge_batch_fk = cb.id
    JOIN screeners scr ON scr.competition_fk = mu.competition_fk
    JOIN screening_challenges sc ON sc.screener_fk = scr.id AND sc.challenge_fk = bc.challenge_fk
    LEFT JOIN batch_challenge_scores bcs ON bcs.batch_challenge_fk = bc.id
    WHERE scr.is_active IS TRUE
    GROUP BY mu.competition_fk, m.id, m.ss58, m.miner_banned_status
),
v_screener_eligible_base_live AS (
    SELECT
        ss.competition_id,
        ss.ss58,
        fs.miner_id,
        fs.script_id,
        ss.avg_score,
        ss.first_upload_at,
        ss.screener_scored,
        rp.screener_required
    FROM v_screener_stats_live ss
    JOIN v_screener_required_pairs_live rp ON rp.competition_id = ss.competition_id
    JOIN v_screener_first_script_live fs ON fs.competition_id = ss.competition_id AND fs.miner_id = ss.miner_id
    JOIN miners m ON m.ss58 = ss.ss58
    WHERE rp.screener_required > 0
      AND ss.screener_scored >= rp.screener_required
      AND m.miner_banned_status IS FALSE
),
v_screener_eligible_ranked_live AS (
    SELECT
        e.competition_id,
        e.ss58,
        e.miner_id,
        e.script_id,
        e.avg_score,
        e.first_upload_at,
        e.screener_scored,
        e.screener_required,
        ROW_NUMBER() OVER (
            PARTITION BY e.competition_id
            ORDER BY e.avg_score DESC NULLS LAST,
                     e.first_upload_at ASC NULLS FIRST,
                     e.ss58 ASC
        ) AS rank,
        COUNT(*) OVER (PARTITION BY e.competition_id) AS total_eligible
    FROM v_screener_eligible_base_live e
),
params AS (
    SELECT COALESCE(CEIL(COUNT(*) * 0.2::numeric)::int, 0) AS top_limit
    FROM v_screener_eligible_base_live r
    WHERE r.competition_id = 40
)
SELECT m.ss58
FROM v_screener_eligible_ranked_live r
JOIN miners m
  ON m.id = r.miner_id
CROSS JOIN params p
WHERE r.competition_id = 40
  AND r.rank <= p.top_limit
  AND m.miner_banned_status IS FALSE
ORDER BY r.rank ASC;


-- =====================================================
-- Optional parity checks vs current v_miner_screener_eligible_ranked
-- =====================================================

-- P1) Count parity
WITH
v_screener_required_pairs_live AS (
    WITH screener_counts AS (
        SELECT s.competition_fk AS competition_id, COUNT(DISTINCT sc.challenge_fk) AS screener_challenge_count
        FROM screeners s
        JOIN screening_challenges sc ON sc.screener_fk = s.id
        WHERE s.is_active IS TRUE
        GROUP BY s.competition_fk
    ),
    ratio_counts AS (
        SELECT cc.competition_fk AS competition_id, COALESCE(json_array_length(ccc.compression_ratios), 1) AS ratio_count
        FROM competition_configs cc
        LEFT JOIN compression_competition_config ccc ON ccc.competition_config_fk = cc.id
    )
    SELECT sc.competition_id, (sc.screener_challenge_count * rc.ratio_count)::bigint AS screener_required
    FROM screener_counts sc
    JOIN ratio_counts rc ON rc.competition_id = sc.competition_id
),
v_screener_first_script_live AS (
    WITH ranked_uploads AS (
        SELECT
            mu.competition_fk AS competition_id,
            s.miner_fk AS miner_id,
            mu.script_fk AS script_id,
            ROW_NUMBER() OVER (PARTITION BY mu.competition_fk, s.miner_fk ORDER BY mu.created_at ASC, mu.script_fk ASC) AS rn
        FROM miner_uploads mu
        JOIN scripts s ON s.id = mu.script_fk
        WHERE mu.competition_fk IS NOT NULL
    )
    SELECT competition_id, miner_id, script_id
    FROM ranked_uploads
    WHERE rn = 1
),
v_screener_stats_live AS (
    SELECT
        mu.competition_fk AS competition_id,
        m.id AS miner_id,
        m.ss58,
        m.miner_banned_status AS is_banned,
        (
            SUM(CASE WHEN bcs.id IS NOT NULL THEN (bcs.score::float / SQRT(bc.compression_ratio::float)) ELSE 0.0 END)
            /
            NULLIF(SUM(CASE WHEN bcs.id IS NOT NULL THEN (1.0 / SQRT(bc.compression_ratio::float)) ELSE 0.0 END), 0)
        ) AS avg_score,
        COUNT(DISTINCT bcs.batch_challenge_fk) AS screener_scored,
        MIN(mu.created_at) AS first_upload_at
    FROM challenge_batches cb
    JOIN scripts s ON s.id = cb.script_fk
    JOIN miners m ON m.id = s.miner_fk
    JOIN miner_uploads mu ON mu.script_fk = s.id
    JOIN batch_challenges bc ON bc.challenge_batch_fk = cb.id
    JOIN screeners scr ON scr.competition_fk = mu.competition_fk
    JOIN screening_challenges sc ON sc.screener_fk = scr.id AND sc.challenge_fk = bc.challenge_fk
    LEFT JOIN batch_challenge_scores bcs ON bcs.batch_challenge_fk = bc.id
    WHERE scr.is_active IS TRUE
    GROUP BY mu.competition_fk, m.id, m.ss58, m.miner_banned_status
),
v_screener_eligible_base_live AS (
    SELECT
        ss.competition_id,
        ss.ss58,
        fs.miner_id,
        fs.script_id,
        ss.avg_score,
        ss.first_upload_at,
        ss.screener_scored,
        rp.screener_required
    FROM v_screener_stats_live ss
    JOIN v_screener_required_pairs_live rp ON rp.competition_id = ss.competition_id
    JOIN v_screener_first_script_live fs ON fs.competition_id = ss.competition_id AND fs.miner_id = ss.miner_id
    JOIN miners m ON m.ss58 = ss.ss58
    WHERE rp.screener_required > 0
      AND ss.screener_scored >= rp.screener_required
      AND m.miner_banned_status IS FALSE
)
SELECT
    (SELECT COUNT(*) FROM v_miner_screener_eligible_ranked r WHERE r.competition_id = 40) AS old_count,
    (SELECT COUNT(*) FROM v_screener_eligible_base_live r WHERE r.competition_id = 40) AS new_count;


-- P2) Top miner_id set parity at top_fraction = 0.2
WITH
v_screener_required_pairs_live AS (
    WITH screener_counts AS (
        SELECT s.competition_fk AS competition_id, COUNT(DISTINCT sc.challenge_fk) AS screener_challenge_count
        FROM screeners s
        JOIN screening_challenges sc ON sc.screener_fk = s.id
        WHERE s.is_active IS TRUE
        GROUP BY s.competition_fk
    ),
    ratio_counts AS (
        SELECT cc.competition_fk AS competition_id, COALESCE(json_array_length(ccc.compression_ratios), 1) AS ratio_count
        FROM competition_configs cc
        LEFT JOIN compression_competition_config ccc ON ccc.competition_config_fk = cc.id
    )
    SELECT sc.competition_id, (sc.screener_challenge_count * rc.ratio_count)::bigint AS screener_required
    FROM screener_counts sc
    JOIN ratio_counts rc ON rc.competition_id = sc.competition_id
),
v_screener_first_script_live AS (
    WITH ranked_uploads AS (
        SELECT
            mu.competition_fk AS competition_id,
            s.miner_fk AS miner_id,
            mu.script_fk AS script_id,
            ROW_NUMBER() OVER (PARTITION BY mu.competition_fk, s.miner_fk ORDER BY mu.created_at ASC, mu.script_fk ASC) AS rn
        FROM miner_uploads mu
        JOIN scripts s ON s.id = mu.script_fk
        WHERE mu.competition_fk IS NOT NULL
    )
    SELECT competition_id, miner_id, script_id
    FROM ranked_uploads
    WHERE rn = 1
),
v_screener_stats_live AS (
    SELECT
        mu.competition_fk AS competition_id,
        m.id AS miner_id,
        m.ss58,
        m.miner_banned_status AS is_banned,
        (
            SUM(CASE WHEN bcs.id IS NOT NULL THEN (bcs.score::float / SQRT(bc.compression_ratio::float)) ELSE 0.0 END)
            /
            NULLIF(SUM(CASE WHEN bcs.id IS NOT NULL THEN (1.0 / SQRT(bc.compression_ratio::float)) ELSE 0.0 END), 0)
        ) AS avg_score,
        COUNT(DISTINCT bcs.batch_challenge_fk) AS screener_scored,
        MIN(mu.created_at) AS first_upload_at
    FROM challenge_batches cb
    JOIN scripts s ON s.id = cb.script_fk
    JOIN miners m ON m.id = s.miner_fk
    JOIN miner_uploads mu ON mu.script_fk = s.id
    JOIN batch_challenges bc ON bc.challenge_batch_fk = cb.id
    JOIN screeners scr ON scr.competition_fk = mu.competition_fk
    JOIN screening_challenges sc ON sc.screener_fk = scr.id AND sc.challenge_fk = bc.challenge_fk
    LEFT JOIN batch_challenge_scores bcs ON bcs.batch_challenge_fk = bc.id
    WHERE scr.is_active IS TRUE
    GROUP BY mu.competition_fk, m.id, m.ss58, m.miner_banned_status
),
v_screener_eligible_base_live AS (
    SELECT
        ss.competition_id,
        ss.ss58,
        fs.miner_id,
        fs.script_id,
        ss.avg_score,
        ss.first_upload_at,
        ss.screener_scored,
        rp.screener_required
    FROM v_screener_stats_live ss
    JOIN v_screener_required_pairs_live rp ON rp.competition_id = ss.competition_id
    JOIN v_screener_first_script_live fs ON fs.competition_id = ss.competition_id AND fs.miner_id = ss.miner_id
    JOIN miners m ON m.ss58 = ss.ss58
    WHERE rp.screener_required > 0
      AND ss.screener_scored >= rp.screener_required
      AND m.miner_banned_status IS FALSE
),
v_screener_eligible_ranked_live AS (
    SELECT
        e.competition_id,
        e.ss58,
        e.miner_id,
        e.script_id,
        e.avg_score,
        e.first_upload_at,
        e.screener_scored,
        e.screener_required,
        ROW_NUMBER() OVER (
            PARTITION BY e.competition_id
            ORDER BY e.avg_score DESC NULLS LAST,
                     e.first_upload_at ASC NULLS FIRST,
                     e.ss58 ASC
        ) AS rank,
        COUNT(*) OVER (PARTITION BY e.competition_id) AS total_eligible
    FROM v_screener_eligible_base_live e
),
limits AS (
    SELECT COALESCE(
        CEIL((SELECT COUNT(*) FROM v_miner_screener_eligible_ranked WHERE competition_id = 40) * 0.2::numeric)::int,
        0
    ) AS top_limit
),
old_top AS (
    SELECT miner_id
    FROM v_miner_screener_eligible_ranked
    WHERE competition_id = 40
      AND rank <= (SELECT top_limit FROM limits)
),
new_top AS (
    SELECT miner_id
    FROM v_screener_eligible_ranked_live
    WHERE competition_id = 40
      AND rank <= (SELECT top_limit FROM limits)
)
SELECT 'old_minus_new' AS side, miner_id FROM old_top
EXCEPT
SELECT 'old_minus_new' AS side, miner_id FROM new_top
UNION ALL
SELECT 'new_minus_old' AS side, miner_id FROM new_top
EXCEPT
SELECT 'new_minus_old' AS side, miner_id FROM old_top;
