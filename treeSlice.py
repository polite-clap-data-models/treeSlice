import polars as pl
import polars.selectors as cs
import polars_hash as plh

from data_design import (
    unit_test_xx3, unit_test_xx_r20_c24_h4_d4_string
)

dag_ds_open, dag_ds_closed, dag_ds_pipe, dag_ds_void = (
    pl.lit(value=" ╠══➤ ", dtype=pl.String), pl.lit(value=" ╚══➤ ", dtype=pl.String),
    pl.lit(value=" ║   ", dtype=pl.String), pl.lit(value="  ", dtype=pl.String)
)
cstart, cstop, chead, ctail, cdepth = "cStart", "cStop", "cHead", "cTail", "cDepth"
hk_ptr, hk_schema, hk_nspace, hk_kvp_treekey = "hkPTR", "hkSchema", "hkNSpace", "hkKVP_TreeKey"
cleft, cright, cis_root, cis_leaf = "cLeft", "cRight", "cIsRoot", "cIsLeaf"
csize, cwidth, cout, cpath = "cSize", "cWidth", "cOut", "cPath"

_sys_dtype = pl.DataFrame(schema={cstart: pl.UInt32}).schema[cstart]
cs_ptr = cs.matches(pattern="^p\\d{3}_[1-9]\\d*$")
cs_vis = cs.matches(pattern="^v\\d{3}_[1-9]\\d*$")
cs_mini = cs.by_name(cdepth, cstart, cstop, chead, ctail, require_all=True)
cs_hash = cs.by_name(hk_ptr, hk_schema, hk_nspace, hk_kvp_treekey, require_all=False)
cs_uqu = (
    cs.starts_with("uq") &
    cs.by_dtype(pl.Struct({"cIsLCA": pl.Boolean, "cIsUQE": pl.Boolean, "cPTR": pl.List(_sys_dtype)}))
)


def unpack_cpath(ingest: pl.DataFrame, /) -> pl.DataFrame:
    assert not ingest.filter(pl.col(cpath).list.contains(None)).height
    ingest: pl.DataFrame = (
        ingest
        .drop(cs_ptr)
        .with_columns(
            pl.col(cpath).list.to_struct(
                n_field_strategy="max_width",
                fields=lambda depth: f"xWS{depth:03}"
            ).alias("yWS")
        )
        .unnest(columns="yWS")
        .drop(cpath)
    )

    go_up = range(len(cs.expand_selector(ingest, cs.starts_with("xWS"))) - 1, - 1, - 1)
    return ingest.rename(
        {f"xWS{gup:03}": f"p{gup:03}_{max(go_up)}" for gup in go_up}
    ).rechunk()


def show_df(ingest: pl.DataFrame, /, *, label: str, verbose: bool = True) -> pl.DataFrame:

    print()

    if verbose:
        # show summary table
        print(f"{label=}  {pl.concat(how='diagonal_relaxed', items=[
            pl.DataFrame({"statistic": ["dtype"]}).hstack(pl.DataFrame({kk: repr(vv)[:100] for kk, vv in ingest.schema.items()})),
            pl.DataFrame({"statistic": ["distinct"]}).hstack(ingest.select(pl.all().n_unique())),
            ingest.describe(),
            # show LC(5) least common five values
            pl.concat(how="horizontal", items=[
                ingest.select(pl.col(col).value_counts(sort=True)).tail(n=5)
                for col in cs.expand_selector(ingest, cs_ptr | cs.by_name(chead, ctail, require_all=False) | cs_hash)
            ]).with_columns(pl.lit("LC(5)").alias("statistic"))
            # show LC(5) least common five values
        ])}")

    # show base table
    print(f"{label=}  {ingest}")

    return ingest


def load_tree(ingest: str | pl.DataFrame | pl.LazyFrame, /, *, n_rows: int | None = None) -> pl.DataFrame:
    return (
        pl.read_parquet(ingest, rechunk=True, n_rows=n_rows)
        if isinstance(ingest, str) and (n_rows or n_rows == 0) else
        pl.read_parquet(ingest, rechunk=True)
        if isinstance(ingest, str) else
        ingest.collect()
        if isinstance(ingest, pl.LazyFrame) else
        ingest
    )


def read_me_3_ways_to_slice() -> bool:

    assert cstart   # used in BOTH a Python AND Polars slice
    assert cstop    # used in a Python slice
    assert csize    # used in a Polars slice
    assert cleft    # used in a SQL slice
    assert cright   # used in a SQL slice

    tree_df: pl.DataFrame = (
        load_tree(unit_test_aa_r52529_c36_h5_d10_string)
        .select(~cs_ptr)
        .select(pl.all().exclude(cout, cwidth, cis_root, cis_leaf))
    )

    assert tuple(col.name for col in tree_df) == tuple(tree_df.schema.keys()) == tuple(tree_df.columns)
    # Polars iterates over a DataFrame by columns, because Polars is columnar
    assert tree_df[4:9].height == 5
    # BUT if you do a Python slice on a Polars DataFrame you'll get ROWS - not COLUMNS
    assert tree_df[4:9].equals(tree_df.slice(offset=4, length=5))
    # Eventhough Polars already has a dedicated row-wise slice feature - this is weird to me

    tree_df.slice(offset=5, length=1).pipe(show_df, label="unit_test_aa_r52529_c36_h5_d10_string.slice(offset=5, length=1)", verbose=False)
    """
    label='unit_test_aa_r52529_c36_h5_d10_string.slice(offset=5, length=1)'  shape: (1, 18)
    ┌───────┬───────┬───────┬───────┬────────┬───────┬───────┬───────┬───────┬───────┬────────┬────────┬──────────┬────────┬────────┬─────────┬───────┬───────┐
    │ cD000 ┆ cD001 ┆ cD002 ┆ cD003 ┆ cD004  ┆ cD005 ┆ cD006 ┆ cD007 ┆ cD008 ┆ cD009 ┆ cDepth ┆ cSize2 ┆ cStartP2 ┆ cStopP ┆ cLeftP ┆ cRightP ┆ cRoot ┆ cSelf │
    │ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---    ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---    ┆ ---    ┆ ---      ┆ ---    ┆ ---    ┆ ---     ┆ ---   ┆ ---   │
    │ str   ┆ str   ┆ str   ┆ str   ┆ str    ┆ str   ┆ str   ┆ str   ┆ str   ┆ str   ┆ u32    ┆ u32    ┆ u32      ┆ u32    ┆ u32    ┆ u32     ┆ str   ┆ str   │
    ╞═══════╪═══════╪═══════╪═══════╪════════╪═══════╪═══════╪═══════╪═══════╪═══════╪════════╪════════╪══════════╪════════╪════════╪═════════╪═══════╪═══════╡
    │  ║    ┆  ║    ┆  ║    ┆       ┆  ╚══➤  ┆ xQ    ┆       ┆       ┆       ┆       ┆ 5      ┆ 798    ┆ 5        ┆ 803    ┆ 6      ┆ 1_601   ┆ bP    ┆ xQ    │
    └───────┴───────┴───────┴───────┴────────┴───────┴───────┴───────┴───────┴───────┴────────┴────────┴──────────┴────────┴────────┴─────────┴───────┴───────┘
    """
    assert len(tree_df[5:803]) == tree_df.slice(offset=5, length=798).height == 798
    tree_df[5].equals(tree_df.slice(offset=5, length=1))
    tree_df[5:803].equals(tree_df.slice(offset=5, length=798))

    pass

    # There are three techniques to get a row-wise **slice** of a table/df:
    #
    # 1. using standard Python methods - phrased with **Absolute** predicates
    #     range(cstart, cstop[, step])
    #     slice(start=cstart, stop=cstop, step=None)
    #     [cstart:cstop:step]
    #
    # 2. using Polars native methods - phrased with **Absolute** predicates
    #     DataFrame.slice(offset=cstart, length=csize)
    #     Series.slice(offset=cstart, length=csize)
    #
    # 3. using SQL range queries (non-equi join, inequality predicates) - phrased with **Relative** predicates
    #     FROM xx AS child, xx AS parent
    #     WHERE child.cleft BETWEEN parent.cleft AND parent.cright
    #
    #     FROM xx AS child, xx AS parent
    #     WHERE child.cstart > parent.cstart AND <= parent.cstop
    #
    #     xx
    #     .join_where(
    #         # 3. using SQL range queries (non-equi join, inequality predicates)
    #         # phrased with **Relative** predicates
    #         xx.select(pl.col(cleft).name.suffix("_child")),
    #         pl.col(cleft + "_child") > pl.col(cleft), pl.col(cleft + "_child") < pl.col(cright)
    #     )
    #
    #     xx
    #     .join_where(
    #         # 3. using SQL range queries (non-equi join, inequality predicates)
    #         # phrased with **Relative** predicates
    #         xx.select(pl.col(cstart).name.suffix("_child")),
    #         pl.col(cstart + "_child") > pl.col(cstart), pl.col(cstart + "_child") <= pl.col(cstop)
    #     )

    tree_df.filter(pl.col(ctail).eq("dX")).pipe(show_df, label="all 25 dX nodes", verbose=False)

    import duckdb as db

    duckdb_query = f"""
    SELECT * FROM tree_df as c
    WHERE {cleft} IN (
        SELECT c.{cleft} FROM tree_df as p
        WHERE c.{cleft} BETWEEN p.{cleft} and p.{cright}
        AND p.{ctail}='dX' and p.{csize} BETWEEN 5 AND 15
    )
    ORDER BY {cleft} ASC
    ;"""

    query_subtrees: tuple[pl.DataFrame, ...] = (

        # The three techniques are functionally equivalent - their output is identical.
        # The 3rd technique is phrased using **Relative** predicates. Whereas techniques 1 and 2
        # are phrased with **Absolute** predicates.

        db.sql(duckdb_query).pl()
        .pipe(show_df, label="subtrees of dX, with size between 5 and 15", verbose=False),

        pl.concat(items=[
            tree_df[19_738: 19_745],
            tree_df[28_516: 28_523]
        ], how="vertical"),

        pl.concat(items=[
            tree_df.slice(offset=19_738, length=7),
            tree_df.slice(offset=28_516, length=7)
        ], how="vertical"),

        tree_df.lazy().set_sorted(cleft)
        .filter(pl.col(cleft).is_in(

            tree_df.lazy().set_sorted(cleft)
            .filter(
                pl.col(ctail).eq("dX") &
                pl.col(csize).is_between(5, 15)
            )
            .join_where(
                tree_df.lazy().set_sorted(cleft).select(cleft, cright),
                pl.col(cleft + "_right") >= pl.col(cleft),
                pl.col(cleft + "_right") < pl.col(cright)
            )
            .collect()
            .get_column(cleft + "_right")

        ))
        .collect(),

        tree_df.lazy().set_sorted(cstart)
        .filter(pl.col(cstart).is_in(

            tree_df.lazy().set_sorted(cstart)
            .filter(
                pl.col(ctail).eq("dX") &
                pl.col(csize).is_between(5, 15)
            )
            .join_where(
                tree_df.lazy().set_sorted(cstart).select(cstart, cstop),
                pl.col(cstart + "_right") >= pl.col(cstart),
                pl.col(cstop + "_right") <= pl.col(cstop)
            )
            .collect()
            .get_column(cstart + "_right")

        ))
        .collect()
    )
    assert (
        query_subtrees[0].height and
        query_subtrees[0].equals(query_subtrees[1]) and
        query_subtrees[0].equals(query_subtrees[2]) and
        query_subtrees[0].equals(query_subtrees[3]) and
        query_subtrees[0].equals(query_subtrees[4])
    )

    """
    label='subtrees of dX, with size between 5 and 15'  shape: (14, 18)
    ┌───────┬───────┬───────┬───────┬───────┬────────┬────────┬────────┬────────┬───────┬────────┬────────┬──────────┬────────┬────────┬─────────┬───────┬───────┐
    │ cD000 ┆ cD001 ┆ cD002 ┆ cD003 ┆ cD004 ┆ cD005  ┆ cD006  ┆ cD007  ┆ cD008  ┆ cD009 ┆ cDepth ┆ cSize2 ┆ cStartP2 ┆ cStopP ┆ cLeftP ┆ cRightP ┆ cRoot ┆ cSelf │
    │ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---    ┆ ---    ┆ ---    ┆ ---    ┆ ---   ┆ ---    ┆ ---    ┆ ---      ┆ ---    ┆ ---    ┆ ---     ┆ ---   ┆ ---   │
    │ str   ┆ str   ┆ str   ┆ str   ┆ str   ┆ str    ┆ str    ┆ str    ┆ str    ┆ str   ┆ u32    ┆ u32    ┆ u32      ┆ u32    ┆ u32    ┆ u32     ┆ str   ┆ str   │
    ╞═══════╪═══════╪═══════╪═══════╪═══════╪════════╪════════╪════════╪════════╪═══════╪════════╪════════╪══════════╪════════╪════════╪═════════╪═══════╪═══════╡
    │       ┆       ┆       ┆       ┆       ┆  ╠══➤  ┆ dX     ┆        ┆        ┆       ┆ 6      ┆ 7      ┆ 19_738   ┆ 19_745 ┆ 39_471 ┆ 39_484  ┆ tL    ┆ dX    │
    │       ┆       ┆       ┆       ┆       ┆  ║     ┆  ╠══➤  ┆ oI     ┆        ┆       ┆ 7      ┆ 3      ┆ 19_739   ┆ 19_742 ┆ 39_472 ┆ 39_477  ┆ tL    ┆ oI    │
    │       ┆       ┆       ┆       ┆       ┆  ║     ┆  ║     ┆  ╚══➤  ┆ iP     ┆       ┆ 8      ┆ 2      ┆ 19_740   ┆ 19_742 ┆ 39_473 ┆ 39_476  ┆ tL    ┆ iP    │
    │       ┆       ┆       ┆       ┆       ┆  ║     ┆  ║     ┆        ┆  ╚══➤  ┆ iS    ┆ 9      ┆ 1      ┆ 19_741   ┆ 19_742 ┆ 39_474 ┆ 39_475  ┆ tL    ┆ iS    │
    │       ┆       ┆       ┆       ┆       ┆  ║     ┆  ╚══➤  ┆ mJ     ┆        ┆       ┆ 7      ┆ 3      ┆ 19_742   ┆ 19_745 ┆ 39_478 ┆ 39_483  ┆ tL    ┆ mJ    │
    │       ┆       ┆       ┆       ┆       ┆  ║     ┆        ┆  ╚══➤  ┆ zJ     ┆       ┆ 8      ┆ 2      ┆ 19_743   ┆ 19_745 ┆ 39_479 ┆ 39_482  ┆ tL    ┆ zJ    │
    │       ┆       ┆       ┆       ┆       ┆  ║     ┆        ┆        ┆  ╚══➤  ┆ zP    ┆ 9      ┆ 1      ┆ 19_744   ┆ 19_745 ┆ 39_480 ┆ 39_481  ┆ tL    ┆ zP    │
    │       ┆  ║    ┆       ┆       ┆       ┆  ╠══➤  ┆ dX     ┆        ┆        ┆       ┆ 6      ┆ 7      ┆ 28_516   ┆ 28_523 ┆ 57_027 ┆ 57_040  ┆ vY    ┆ dX    │
    │       ┆  ║    ┆       ┆       ┆       ┆  ║     ┆  ╠══➤  ┆ vP     ┆        ┆       ┆ 7      ┆ 3      ┆ 28_517   ┆ 28_520 ┆ 57_028 ┆ 57_033  ┆ vY    ┆ vP    │
    │       ┆  ║    ┆       ┆       ┆       ┆  ║     ┆  ║     ┆  ╚══➤  ┆ vZ     ┆       ┆ 8      ┆ 2      ┆ 28_518   ┆ 28_520 ┆ 57_029 ┆ 57_032  ┆ vY    ┆ vZ    │
    │       ┆  ║    ┆       ┆       ┆       ┆  ║     ┆  ║     ┆        ┆  ╚══➤  ┆ oP    ┆ 9      ┆ 1      ┆ 28_519   ┆ 28_520 ┆ 57_030 ┆ 57_031  ┆ vY    ┆ oP    │
    │       ┆  ║    ┆       ┆       ┆       ┆  ║     ┆  ╚══➤  ┆ xO     ┆        ┆       ┆ 7      ┆ 3      ┆ 28_520   ┆ 28_523 ┆ 57_034 ┆ 57_039  ┆ vY    ┆ xO    │
    │       ┆  ║    ┆       ┆       ┆       ┆  ║     ┆        ┆  ╚══➤  ┆ mG     ┆       ┆ 8      ┆ 2      ┆ 28_521   ┆ 28_523 ┆ 57_035 ┆ 57_038  ┆ vY    ┆ mG    │
    │       ┆  ║    ┆       ┆       ┆       ┆  ║     ┆        ┆        ┆  ╚══➤  ┆ xD    ┆ 9      ┆ 1      ┆ 28_522   ┆ 28_523 ┆ 57_036 ┆ 57_037  ┆ vY    ┆ xD    │
    └───────┴───────┴───────┴───────┴───────┴────────┴────────┴────────┴────────┴───────┴────────┴────────┴──────────┴────────┴────────┴─────────┴───────┴───────┘
    """

    return True


def use_enums(ingest: pl.DataFrame, /) -> pl.DataFrame:
    if set(ingest.select(cs_ptr | cs.by_name(chead, ctail, require_all=False)).schema.values()) == {pl.String}:
        tree_namespace = pl.Enum(ingest.get_column(ctail).unique())
        # polars.Enum, polars.Categorical
        for idx, col in enumerate(ingest.columns):
            if col in cs.expand_selector(ingest, cs_ptr | cs.by_name(chead, ctail, require_all=False)):
                ingest.replace_column(index=idx, column=ingest.get_column(col).cast(
                    dtype=tree_namespace, strict=True
                ).rechunk(in_place=True))
    return ingest


def gen_hash_keys(ingest: pl.DataFrame, /) -> pl.DataFrame:

    sep, sqs, treat_quotes_and_whitespace_zls = "|", "'", (r"""[\s"'`]""", "")
    hk_salt: dict = {
        hk_schema: f"{sep}{ingest.schema[cstart]}".replace(sqs, sep),
        hk_nspace: f"{sep}{ingest.schema[ctail]}".replace(sqs, sep),
        hk_kvp_treekey: f"{sep}{ingest.schema[cdepth]}{sep}{ingest.schema[ctail]}".replace(sqs, sep)
    }
    hk_schema_leaf, hk_nspace_leaf = [
        pl.lit(zz) for zz in
        pl.DataFrame({"yy": [hk_salt[hk_schema], hk_salt[hk_nspace]]})
        .select(plh.col("*").nchash.wyhash()
        .map_batches(return_dtype=pl.Utf8, function=lambda ss: pl.Series(
            "_".join((f"{xx:016x}"[:4], f"{xx:016x}"[4:-4], f"{xx:016x}"[-4:])) for xx in ss
        ))).get_column("yy").to_list()
    ]

    trees = pl.concat(how="vertical", items=(
        ingest.set_sorted(cstart)
        .slice(offset=_indv_tree[cstart], length=_indv_tree[cstop]-_indv_tree[cstart])
        .select(
            pl.lit(dtype=_sys_dtype, value=_indv_tree[cstart]).alias(cstart),
            (pl.concat_str([cdepth, ctail, pl.lit(sep)], separator=sep).str.join() + hk_salt[hk_kvp_treekey]).nchash.wyhash()
            .map_batches(return_dtype=pl.Utf8, function=lambda ss: pl.Series(
                "_".join((f"{xx:016x}"[:4], f"{xx:016x}"[4:-4], f"{xx:016x}"[-4:])) for xx in ss
            )).alias(hk_kvp_treekey)
        ) for _indv_tree in ingest.set_sorted(cstart).filter(pl.col(cdepth).eq(0)).select(cstart, cstop).to_dicts()
    )).rechunk()

    self_range_join_no_leaves: pl.LazyFrame = (
        #
        # This ↓↓↓ is a Polars version of a standard SQL self-join range query (in Open form). It finds each
        # Tree node's Progeny using **Relative** predicates. A more familiar SQL equivalent might be
        # `FROM t AS child, t AS parent WHERE child.cstart > parent.cstart AND child.cstart < parent.cstop;`
        #
        # Note, we are using lazy evaluation. Pay attention to the set_sorted expressions. The nodes are collocated
        # (ordered) according to the **Relative** (Nested Set) predicates we are using (cstart, cstop). Pay attention
        # to the filters. The Progenitor side drops Leaves, whereas the Progeny side drops Roots.
        ingest
        .lazy().set_sorted(cstart)
        .filter(~pl.col(cstop).sub(cstart).eq(1))
        .select(pl.col(cstart, cstop).name.prefix("Progenitor_"))
        # **Tree Progenitor ↑↑↑ (Parent, Root)**
        # - Means ancestor or parent; it is the original source or originator of a lineage from which others
        #   descend.
        #
        .join_where(
            ingest.lazy().set_sorted(cstart)
            .filter(~pl.col(cdepth).eq(0))
            .select(
                pl.col(cstart).name.prefix("Progeny_"),
                pl.col(ctail).cast(dtype=pl.Utf8, strict=True)
                .str.to_lowercase().str.replace_all(*treat_quotes_and_whitespace_zls)
                .alias(hk_nspace),
                pl.col(cstop).sub(cstart).cast(dtype=pl.Utf8, strict=True)
                .alias(hk_schema)
            ),
            # **Tree Progeny ↑↑↑ (Children, Subtree)**
            # - Refers to the offspring or descendants of a person, animal, or plant, essentially denoting the next
            #   generation or results of reproduction.
            #
            pl.col("Progeny_" + cstart) > pl.col("Progenitor_" + cstart),
            pl.col("Progeny_" + cstart) < pl.col("Progenitor_" + cstop)
            #
            # Most **Relative** (Nested Set) queries you'll encounter use the `Closed` subtree query form. That approach
            # includes both the Progenitor and Progeny. In that form, the `self` row is inherently valid. Therefore, any
            # in-scope node always includes itself, ensuring non-null and non-empty query results.
            #
            # **Open Form (Exclusive range query)**
            # - This excludes boundary values. For instance, `WHERE xyz > 5 AND xyz < 9` may return 6 through 8
            #   only.
            #
            # **Closed Form (Inclusive range query)**
            # - The `BETWEEN` predicate includes boundary values. For instance, `WHERE xyz BETWEEN 5 AND 9` may
            #   return 5 through 9. You get the same outcome with `>=` and `<=` as well.
            #
        )
        .group_by([pl.col("Progenitor_" + cstart)], maintain_order=False)
        .agg(hk_schema, hk_nspace)
        # This agg expression ↑↑↑ constructs two Progeny list[string] columns
        # - at this point the two columns are NON-NULL and the lists are both NON-EMPTY and NON-NULL
        # - each has the full Progeny (as strings) in a single row (as a list)
        # - we have two for different contexts
        # - Leaf Nodes are absent
        #
        .select(
            pl.col("Progenitor_" + cstart).alias(cstart),

            (
                (pl.col(hk_schema).list.sort().list.join(sep) + hk_salt[hk_schema])
                .nchash.wyhash()
                .map_batches(return_dtype=pl.Utf8, function=lambda ss: pl.Series(
                "_".join((f"{xx:016x}"[:4], f"{xx:016x}"[4:-4], f"{xx:016x}"[-4:])) for xx in ss
            ))
            ).name.keep(),

            (
                (pl.col(hk_nspace).list.unique(maintain_order=False).list.sort().list.join(sep) + hk_salt[hk_nspace])
                .nchash.wyhash()
                .map_batches(return_dtype=pl.Utf8, function=lambda ss: pl.Series(
                "_".join((f"{xx:016x}"[:4], f"{xx:016x}"[4:-4], f"{xx:016x}"[-4:])) for xx in ss
            ))
            ).name.keep()
        )
    )

    return (
        ingest
        .lazy().set_sorted(cstart)
        .join(on=cstart, how="left", allow_parallel=True, other=self_range_join_no_leaves)
        .join(on=cstart, how="left", allow_parallel=True, other=trees.lazy())
        # These joins ↑↑↑ are phrased with an **Absolute** predicate.
        # - self_range_join_no_leaves does not carry Leaf Nodes.
        # - it does this because it's an efficient approach, since a Leaf is (for our purposes) a known/constant value.
        # - this join will materialise the missing Leaf Nodes as NULL.
        #
        .with_columns(
            pl.coalesce(pl.col(hk_schema), hk_schema_leaf).name.keep(),
            pl.coalesce(pl.col(hk_nspace), hk_nspace_leaf).name.keep(),
            pl.col(hk_kvp_treekey).fill_null(strategy="forward")
        )
    ).collect()


def build_trees(ingest: pl.DataFrame, /) -> pl.LazyFrame:
    """
    Computes integer-based boundaries and other metrics for each node in a tree DataFrame.
    In addition to assigning left and right pointers for nested set style queries, this function
    calculates values like depth, out-degree, and subtree size. These columns enable efficient
    subtree extraction in both Python (through slicing) and SQL (using BETWEEN clauses).
    By enriching the DataFrame with these bounds and metrics, downstream operations on hierarchical
    data become more flexible, performant, and intuitive.
    """
    #
    # Best measured performance with default parameters is ≈ 11.3M rows/sec. The system processes 100,000 rows
    # in ≈ 8.85 ms, which is only 1.77% of the original 500 ms performance budget. This indicates exceptional
    # efficiency, with 98.23% of the target window unused, leaving ample headroom for additional workload.
    #
    ptr_key_cols: tuple[str, ...] = tuple(sorted(cs.expand_selector(ingest, cs_ptr)))
    if ptr_key_cols:
        assert tuple(cs.expand_selector(ingest, cs_ptr)) == ptr_key_cols
        go_up = range(len(ptr_key_cols) - 1, - 1, - 1)

        sep, sqs, treat_quotes_and_whitespace_zls = "|", "'", (r"""[\s"'`]""", "")
        hk_salt: dict = {
            hk_ptr: f"{sep}{ingest.schema[ptr_key_cols[0]]}".replace(sqs, sep)
        }

        for gup in go_up:
            # going from leaves up(↑) to the roots [..., 3, 2, 1, 0] descending integers
            if gup == max(go_up):
                build_working_storage = ()
                is_aligned = pl.col(ptr_key_cols[gup]).is_not_null()
                build_ctail = pl.when(is_aligned).then(ptr_key_cols[gup])
                build_cdepth = pl.when(is_aligned).then(pl.lit(gup, dtype=_sys_dtype))
                build_cstop = pl.when(is_aligned).then(pl.lit(1))
                continue
            build_working_storage += (pl.struct(pl.col(ptr_key_cols[gup])).rle_id().set_sorted().alias(f"xWS{gup:03}"),)
            is_aligned = pl.col(ptr_key_cols[gup]).is_not_null() & pl.col(ptr_key_cols[gup + 1]).is_null()
            build_ctail = build_ctail.when(is_aligned).then(ptr_key_cols[gup])
            build_cdepth = build_cdepth.when(is_aligned).then(pl.lit(gup, dtype=_sys_dtype))
            build_cstop = build_cstop.when(is_aligned).then(pl.col(f"xWS{gup:03}").len().over(f"xWS{gup:03}"))
            # build_cstop = build_cstop.when(is_aligned).then(
            #     # correct result, looks compelling, but only for super SHALLOW WIDE TREES
            #     # +7.81% cpu: the cost of the test far outweighs the benefit
            #     pl.when(pl.col(ptr_key_cols[gup + 1]).shift(-1).is_null()).then(
            #         pl.lit(1)
            #     ).otherwise(
            #         pl.col(f"xWS{gup:03}").len().over(f"xWS{gup:03}")
            #     )
            # )

        ingest: pl.LazyFrame = (
            ingest
            .select(ptr_key_cols)
            .lazy()
            .with_columns(
                pl.col(ptr_key_cols[0]).alias(chead),
                build_ctail.alias(ctail),
                build_cdepth.alias(cdepth),
                pl.int_range(pl.len(), dtype=_sys_dtype).set_sorted().alias(cstart),
                *build_working_storage
            )
            .with_columns(
                build_cstop.add(pl.col(cstart)).alias(cstop)
            )
        )

        del build_working_storage, build_ctail, build_cdepth, build_cstop
        for gup in go_up:
            # going from leaves up(↑) to the roots [..., 3, 2, 1, 0] descending integers

            if gup == max(go_up):
                is_aligned = pl.col(ptr_key_cols[gup]).is_not_null()
                build_cwidth = pl.when(is_aligned).then(pl.lit(True).filter(is_aligned).len())
                build_cout = pl.when(is_aligned).then(0)
                continue
            is_aligned = pl.col(ptr_key_cols[gup]).is_not_null() & pl.col(ptr_key_cols[gup + 1]).is_null()
            build_cwidth = build_cwidth.when(is_aligned).then(pl.lit(True).filter(is_aligned).len())
            build_cout = build_cout.when(is_aligned).then(
                (
                    pl.col(f"xWS{gup:03}")
                    if gup == max(go_up) - 1 else
                    # # correct result, looks compelling, but defintely NOT
                    # # +200.00% cpu
                    # pl.col(f"xWS{gup + 1:03}").unique()
                    pl.col(f"xWS{gup:03}").filter(pl.col(ptr_key_cols[gup + 2]).is_null())
                ).len().sub(1).over(f"xWS{gup:03}")
            )

        # `pyo3_runtime.PanicException: Polars' maximum length reached. Consider installing 'polars-u64-idx'.
        # This error is caused by our build_hk_schema and build_hk_nspace implementation over too-large
        # a volume. We kept this logic seperate in the gen_hash_keys which is slower but more reliable.
        #
        # The error is raised when Polars’ internal indexing limit is exceeded. By default,
        # Polars uses a 32‑bit index (or similar constrained data type), which means it can only
        # address up to a certain number of rows or elements. When you work with an extremely
        # large DataFrame or perform operations that inadvertently produce an enormous number of
        # rows (or an intermediate result that explodes in size), you may hit this limit. The
        # message suggests installing **polars-u64-idx**, which switches the underlying index
        # to a 64‑bit variant, allowing for a much larger maximum length.

        build_cvis_a = tuple(
            pl.when(pl.col(cdepth).eq(gup)).then(pl.col(cstop))
            .fill_null(strategy="forward")
            .alias(f"v{gup:03}_{max(go_up)}")
            for gup in go_up
        )

        build_cvis_b = tuple((
            pl
            .when(pl.col(cdepth).lt(gup)).then(dag_ds_void)
            .when(pl.col(cdepth).ge(gup + 2)).then(
                pl.when(pl.col(f"v{gup:03}_{max(go_up)}").eq(pl.col(f"v{gup + 1:03}_{max(go_up)}")))
                .then(dag_ds_void).otherwise(dag_ds_pipe)
            )
            .when(pl.col(cdepth).eq(gup + 1)).then(
                pl.when(pl.col(f"v{gup:03}_{max(go_up)}").eq(pl.col(f"v{gup + 1:03}_{max(go_up)}")))
                .then(dag_ds_closed).otherwise(dag_ds_open)
            )
            .otherwise(
                pl.col(ctail).cast(dtype=pl.Utf8)
                .str.strip_chars().str.slice(offset=0, length=9)
            )
            if gup < max(go_up) - 1 else

            pl
            .when(pl.col(cdepth).lt(gup)).then(dag_ds_void)
            .when(pl.col(cdepth).eq(gup + 1)).then(
                pl.when(pl.col(f"v{gup:03}_{max(go_up)}").eq(pl.col(f"v{gup + 1:03}_{max(go_up)}")))
                .then(dag_ds_closed).otherwise(dag_ds_open)
            )
            .otherwise(
                pl.col(ctail).cast(dtype=pl.Utf8)
                .str.strip_chars().str.slice(offset=0, length=9)
            )
            if gup < max(go_up) else

            pl
            .when(pl.col(cdepth).lt(gup)).then(dag_ds_void)
            .otherwise(
                pl.col(ctail).cast(dtype=pl.Utf8)
                .str.strip_chars().str.slice(offset=0, length=9)
            )

        ).alias(f"v{gup:03}_{max(go_up)}") for gup in go_up)

        return (
            ingest
            .with_columns(
                # +88.72% combined cpu cost ↓↓↓
                build_cwidth.alias(cwidth),
                build_cout.alias(cout),
                # +88.72% combined cpu cost ↑↑↑

                # +3.86% combined cpu cost ↓↓↓
                pl.col(cstop).sub(cstart).alias(csize),
                pl.col(cdepth).eq(0).alias(cis_root),
                pl.col(cstop).sub(cstart).eq(1).alias(cis_leaf),
                pl.col(cstart).mul(2).sub(cdepth).add(1).cast(dtype=_sys_dtype).alias(cleft),
                pl.col(cstop).mul(2).sub(cdepth).cast(dtype=_sys_dtype).alias(cright),
                # +3.86% combined cpu cost ↑↑↑
                *build_cvis_a[::-1],

                (pl.concat_str(pl.col(ptr_key_cols), ignore_nulls=True, separator=sep) + hk_salt[hk_ptr])
                .nchash.wyhash()
                .map_batches(return_dtype=pl.Utf8, function=lambda ss: pl.Series(
                    "_".join((f"{xx:016x}"[:4], f"{xx:016x}"[4:-4], f"{xx:016x}"[-4:])) for xx in ss
                ))
                .alias(hk_ptr)

            )
            .with_columns(*build_cvis_b)
            .select(~(cs.starts_with("xWS")))
        )

    else:
        go_down = range(ingest.get_column(cdepth).max() + 1)
        return (
            ingest.lazy()
            .select(
                pl.when(pl.col(cdepth).lt(gdwn)).then(None)
                .otherwise(
                    pl.when(pl.col(cdepth).ne(gdwn)).then(None)
                    .otherwise(ctail).forward_fill()
                ).alias(f"p{gdwn:03}_{max(go_down)}")
                for gdwn in go_down
            )
            .collect()
            .pipe(build_trees)
        )


def refine_paths(ingest: pl.DataFrame, /) -> pl.DataFrame:
    """
    Refines a DataFrame of tree paths by ensuring completeness and proper ordering.
    It deduplicates rows, co-locates sibling nodes under their shared parent, and
    generates any missing intermediate nodes for leaf-only paths. The result is a
    coherent, fully expanded hierarchy that accurately represents each node's position
    in the tree. Ideal for downstream processing or visualization of hierarchical data.
    """
    ptr_key_cols: tuple[str, ...] = tuple(sorted(cs.expand_selector(ingest, cs_ptr)))

    _helper: pl.DataFrame = pl.DataFrame({
        "xCartProd": [xcp + 1 for xcp in range(len(ptr_key_cols)) for _ in range(xcp, - 1, - 1)],
        "xNulls": [len(ptr_key_cols) - xnull for xcp in range(len(ptr_key_cols)) for xnull in range(xcp, - 1, - 1)]
    })

    going_from_roots_down_to_the_leaves = range(len(ptr_key_cols))
    # going from roots down(↓) to the leaves [0, 1, 2, 3, ...] ascending integer
    # assert list(going_from_roots_down_to_the_leaves)[:3] == [0, 1, 2]

    # go_up = range(len(ptr_key_cols) - 1, - 1, - 1)
    # # going from leaves up(↑) to the roots [..., 3, 2, 1, 0] descending integers
    # # assert list(go_up)[-3:] == [2, 1, 0]

    ingest: pl.DataFrame = (
        ingest
        .lazy()

        .with_columns(
            pl.int_range(pl.len(), dtype=pl.UInt32).set_sorted()
            .alias("xRowNumber")
        )
        .with_columns(
            pl.col("xRowNumber").first().over(ptr_key_cols[:gdown + 1])
            .alias(f"xWS{gdown:03}")
            for gdown in going_from_roots_down_to_the_leaves[:-1]
        )
        .sort([f"xWS{gdown:03}" for gdown in going_from_roots_down_to_the_leaves[:-1]] + ["xRowNumber"])
        # ↑↑↑ step1 co-locate (cluster, arrange) the nodes (rows, paths)

        .with_columns(
            pl.sum_horizontal(
                pl.col("xRowNumber").eq(pl.col(f"xWS{gdown:03}"))
                for gdown in going_from_roots_down_to_the_leaves[:-1]
            )
            .add(1).alias("xCartProd")
        )
        .join(_helper.lazy(), on="xCartProd", how="inner", maintain_order="left")
        .with_columns(
            pl.when(pl.col("xNulls").gt(gdown))
            .then(pl.col(ptr_key_cols[gdown]))
            .otherwise(None)
            .alias(ptr_key_cols[gdown])
            for gdown in going_from_roots_down_to_the_leaves
        )
        # ↑↑↑ step2 explode the nodes (rows, paths)

        .select(cs_ptr)

        .unique(keep="first", maintain_order=True)
        # ↑↑↑ step3 dedupe the nodes (rows, paths) - first arrival is kept

        .collect()
    )

    return ingest


# demonstration
if __name__ == "__main__":

    with pl.Config(
        thousands_separator="_", tbl_width_chars=1_200,
        fmt_str_lengths=80, fmt_table_cell_list_len=20,
        tbl_cols=200, tbl_rows=50
    ):

        (
            load_tree(unit_test_xx_r20_c24_h4_d4_string)
            .pipe(build_trees).collect()
            .pipe(use_enums)
            .pipe(gen_hash_keys)
            .select(~cs_ptr)
            .pipe(show_df, label="unit_test_xx_r20_c24_h4_d4_string", verbose=True)
        )


# unit tests
if __name__ == "XXX __main__":
    from polars.testing import assert_frame_equal

    assert_frame_equal(
        load_tree(unit_test_xx_r20_c24_h4_d4_string).select(cs_ptr).pipe(build_trees).collect().pipe(gen_hash_keys),
        load_tree(unit_test_xx_r20_c24_h4_d4_string),
        check_exact=True, check_row_order=True, check_column_order=True
    )
    assert_frame_equal(
        load_tree(unit_test_xx3).pipe(unpack_cpath).select(cs_ptr).pipe(refine_paths).pipe(build_trees).collect().pipe(gen_hash_keys),
        load_tree(unit_test_xx_r20_c24_h4_d4_string),
        check_exact=True, check_row_order=True, check_column_order=True
    )

    print("NO REGRESSIONS detected")

