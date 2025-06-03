from json import loads
import polars as pl
import polars.selectors as cs
import polars_hash as plh

chead, ctail, cdepth, cstart, cstop = "cHead", "cTail", "cDepth", "cStart", "cStop"
hk_ptr, hk_ptr_parent, hk_schema, hk_nspace, hk_kvp_treekey = "hkPTR", "hkPTR_alParent", "hkSchema", "hkNSpace", "hkKVP_TreeKey"
csize, cwidth, cout = "cSize", "cWidth", "cOut"
cleft, cright, cis_root, cis_leaf = "cLeft", "cRight", "cIsRoot", "cIsLeaf"
cptr, cparent, cchild, cbkey = "cPTR", "cParent", "cChild", "cBKey"
vis_open, vis_closed, vis_pipe, vis_void = (
    pl.lit(value=" ╠══➤ ", dtype=pl.String), pl.lit(value=" ╚══➤ ", dtype=pl.String),
    pl.lit(value=" ║   ", dtype=pl.String), pl.lit(value="  ", dtype=pl.String)
)

_sys_dtype = pl.DataFrame(schema={cstart: pl.UInt32}).schema[cstart]
cs_ptr = cs.matches(pattern="^p\\d{3}_[1-9]\\d*$")
cs_vis = cs.matches(pattern="^v\\d{3}_[1-9]\\d*$")
_core_12 = (chead, ctail, cdepth, cstart, cstop, csize, cwidth, cout, cleft, cright, cis_root, cis_leaf)
cs_core_12 = cs.by_name(*_core_12, require_all=False)
cs_hash = cs.by_name(hk_ptr, hk_ptr_parent, hk_schema, hk_nspace, hk_kvp_treekey, require_all=False)
cs_uqu = (
    cs.starts_with("uq") &
    cs.by_dtype(pl.Struct({"cIsLCA": pl.Boolean, "cIsUQE": pl.Boolean, cptr: pl.List(_sys_dtype)}))
)


def tree_hook(ingest: dict | str | pl.DataFrame | pl.LazyFrame, /) -> dict[str, bool]:

    _schema_hook: dict = {
        "isDICT":  lambda Δ: (None, isinstance(Δ, dict)),
        "isJSON":  lambda Δ: (None, isinstance(Δ, str) and Δ.strip()[0] == "{" and Δ.strip()[-1] == "}" and isinstance(loads(Δ), dict)),  # json nested object

        "isPLFL":  lambda Δ: (None, isinstance(Δ, str) and Δ[-8:] == ".parquet" and (True if pl.scan_parquet(Δ, n_rows=0).collect_schema() else False)),
        "isPL":    lambda Δ: (None, isinstance(Δ, pl.DataFrame | pl.LazyFrame)),

        "mWide":   lambda Δ: (cs.expand_selector(Δ, ~(cs_ptr | cs_vis | cs_core_12 | cs_hash | cs_uqu)), len(cs.expand_selector(Δ, cs_core_12)) == 12 and len(cs.expand_selector(Δ, cs_ptr)) >= 1),  # previously FULL
        "mI05":    lambda Δ: (cs.expand_selector(Δ, ~(cs_ptr | cs_vis | cs_core_12 | cs_hash | cs_uqu)), len(cs.expand_selector(Δ, cs.by_name(chead, ctail, cdepth, cstart, cstop, require_all=False))) == 5),  # previously CS_MINI_05
        "mSlim":   lambda Δ: (cs.expand_selector(Δ, ~(cs_ptr | cs_vis | cs_core_12 | cs_hash | cs_uqu)), len(cs.expand_selector(Δ, cs.by_name(ctail, cdepth, require_all=False))) == 2),  # previously MVP

        "mList":   lambda Δ: (cs.expand_selector(Δ, ~(cs_ptr | cs_vis | cs_core_12 | cs_hash | cs_uqu | cs.by_name(cptr, require_all=False))), len(cs.expand_selector(Δ, cs.by_name(cptr, require_all=False))) == 1 and repr(Δ.lazy().collect_schema()[cptr])[:5] == "List("),
        "mPTR":    lambda Δ: (cs.expand_selector(Δ, ~(cs_ptr | cs_vis | cs_core_12 | cs_hash | cs_uqu)), len(cs.expand_selector(Δ, cs_ptr)) >= 1),

        "mALPC":   lambda Δ: (cs.expand_selector(Δ, ~(cs_ptr | cs_vis | cs_core_12 | cs_hash | cs_uqu | cs.by_name(cparent, cchild, cbkey, require_all=False))), len(cs.expand_selector(Δ, cs.by_name(cparent, cchild, cbkey, require_all=False))) == 3)
    }

    return {kk: _schema_hook[kk](ingest) for kk in _schema_hook}


def use_enums(ingest: pl.DataFrame, /, *, namespace_map: dict[str, str] = None) -> pl.DataFrame:

    if not namespace_map and ingest.schema[ctail] == pl.String:
        namespace_map: dict = {ptr_column: ctail for ptr_column in
            cs.expand_selector(ingest, cs_ptr | cs.by_name(chead, ctail, require_all=True))
        }

    if namespace_map:
        assert all(namespace_map.keys()) and all(namespace_map.values())
        assert all(vv in namespace_map.keys() for vv in namespace_map.values())
        assert all(ingest.schema[kk] == pl.String for kk in namespace_map.keys())
        namespace: dict[str, pl.Enum] = {
            col: pl.Enum(ingest.get_column(col).unique()) for col in namespace_map.values()
        }

        for idx, col in enumerate(ingest.columns):
            if col in namespace_map:
                ingest.replace_column(
                    index=idx, column=ingest.get_column(col).cast(
                    dtype=namespace[namespace_map[col]], strict=True
                ).rechunk(in_place=True))

    return ingest


def gen_hash_keys(ingest: pl.DataFrame, /, *, as_uint64: bool = True) -> pl.DataFrame:

    ptr_key_cols: tuple[str, ...] = tuple(sorted(cs.expand_selector(ingest, cs_ptr)))
    going_from_roots_down_to_the_leaves: range = range(len(ptr_key_cols))
    # going from roots down(↓) to the leaves [0, 1, 2, 3, ...] ascending(↑) integers (cdepth)
    going_from_leaves_up_to_the_roots: range = range(len(ptr_key_cols) - 1, - 1, - 1)
    # going from leaves up(↑) to the roots [..., 3, 2, 1, 0] descending(↓) integers (cdepth)
    assert ptr_key_cols and tuple(cs.expand_selector(ingest, cs_ptr)) == ptr_key_cols

    sep, sqs, treat_quotes_and_whitespace_zls = "|", "'", (r"""[\s"'`]""", "")
    hk_salt: dict = {
        hk_ptr:         f"{sep}{ingest.schema[ctail]}".replace(sqs, sep),
        hk_schema:      f"{sep}{ingest.schema[cstart]}".replace(sqs, sep),
        hk_nspace:      f"{sep}{ingest.schema[ctail]}".replace(sqs, sep),
        hk_kvp_treekey: f"{sep}{ingest.schema[cdepth]}{sep}{ingest.schema[ctail]}".replace(sqs, sep)
    }

    root_nodes: pl.DataFrame = pl.concat(how="vertical", items=(
        ingest.set_sorted(cstart)
        .slice(offset=_indv_tree[cstart], length=_indv_tree[cstop]-_indv_tree[cstart])
        .select(
            pl.lit(dtype=_sys_dtype, value=_indv_tree[cstart]).alias(cstart),
            (pl.concat_str([cdepth, ctail, pl.lit(sep)], separator=sep).str.join() + hk_salt[hk_kvp_treekey])
            .nchash.xxhash64().alias(hk_kvp_treekey)
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

            (pl.col(hk_schema).list.sort().list.join(sep) + hk_salt[hk_schema])
            .nchash.xxhash64().name.keep(),

            (pl.col(hk_nspace).list.unique(maintain_order=False).list.sort().list.join(sep) + hk_salt[hk_nspace])
            .nchash.xxhash64().name.keep()
        )
    )

    return (
        ingest
        .lazy().set_sorted(cstart)
        .with_columns(

            (pl.concat_str(ptr_key_cols, ignore_nulls=True, separator=sep) + hk_salt[hk_ptr])
            .nchash.xxhash64().alias(hk_ptr),

            pl.when(pl.col(cdepth).eq(0)).then(None).otherwise(pl.concat_str(
                (
                    pl.when(pl.col(cdepth).eq(xx)).then(None).otherwise(pl.col(yy))
                    for xx, yy in enumerate(ptr_key_cols)
                ),
                ignore_nulls=True, separator=sep) + hk_salt[hk_ptr]
             )
            .nchash.xxhash64().alias(hk_ptr_parent)

        )
        .join(on=cstart, how="left", allow_parallel=True, other=self_range_join_no_leaves)
        .join(on=cstart, how="left", allow_parallel=True, other=root_nodes.lazy())
        # These joins ↑↑↑ are phrased with an **Absolute** predicate.
        # - self_range_join_no_leaves does not carry Leaf Nodes.
        # - it does this because it's an efficient approach, since a Leaf is (for our purposes) a known/constant value.
        # - this join will materialise the missing Leaf Nodes as NULL.
        #
        .with_columns(
            (pl.col(hk_ptr) if as_uint64 else pl.col(hk_ptr).map_batches(
                return_dtype=pl.Utf8, function=lambda hash_keys: pl.Series(
                "_".join((f"{hk:016x}"[:4], f"{hk:016x}"[4:-4], f"{hk:016x}"[-4:])) if hk else None for hk in hash_keys
            )))
            .name.keep(),

            (pl.col(hk_ptr_parent) if as_uint64 else pl.col(hk_ptr_parent).map_batches(
                return_dtype=pl.Utf8, function=lambda hash_keys: pl.Series(
                "_".join((f"{hk:016x}"[:4], f"{hk:016x}"[4:-4], f"{hk:016x}"[-4:])) if hk else None for hk in hash_keys
            )))
            .name.keep(),

            (pl.col(hk_schema) if as_uint64 else pl.col(hk_schema).map_batches(
                return_dtype=pl.Utf8, function=lambda hash_keys: pl.Series(
                "_".join((f"{hk:016x}"[:4], f"{hk:016x}"[4:-4], f"{hk:016x}"[-4:])) if hk else None for hk in hash_keys
            )))
            .name.keep(),

            (pl.col(hk_nspace) if as_uint64 else pl.col(hk_nspace).map_batches(
                return_dtype=pl.Utf8, function=lambda hash_keys: pl.Series(
                "_".join((f"{hk:016x}"[:4], f"{hk:016x}"[4:-4], f"{hk:016x}"[-4:])) if hk else None for hk in hash_keys
            )))
            .name.keep(),

            (pl.col(hk_kvp_treekey) if as_uint64 else pl.col(hk_kvp_treekey).map_batches(
                return_dtype=pl.Utf8, function=lambda hash_keys: pl.Series(
                "_".join((f"{hk:016x}"[:4], f"{hk:016x}"[4:-4], f"{hk:016x}"[-4:])) if hk else None for hk in hash_keys
            )))
            .fill_null(strategy="forward").name.keep()
        )
    ).collect()


def refine_model(ingest: pl.DataFrame | pl.LazyFrame, /) -> pl.LazyFrame:
    """
    Refines a DataFrame of tree paths by ensuring completeness and proper ordering.
    It deduplicates rows, co-locates sibling nodes under their shared parent, and
    generates any missing intermediate nodes for leaf-only paths. The result is a
    coherent, fully expanded hierarchy that accurately represents each node's position
    in the tree. Ideal for downstream processing or visualisation of hierarchical data.
    """
    # refine_model (co-locate, explode, dedupe)

    ptr_key_cols: tuple[str, ...] = tuple(sorted(cs.expand_selector(ingest, cs_ptr)))
    going_from_roots_down_to_the_leaves: range = range(len(ptr_key_cols))
    # going from roots down(↓) to the leaves [0, 1, 2, 3, ...] ascending(↑) integers (cdepth)
    going_from_leaves_up_to_the_roots: range = range(len(ptr_key_cols) - 1, - 1, - 1)
    # going from leaves up(↑) to the roots [..., 3, 2, 1, 0] descending(↓) integers (cdepth)
    assert ptr_key_cols and tuple(cs.expand_selector(ingest, cs_ptr)) == ptr_key_cols

    ws_row_number, ws_cart_prod, ws_nulls = "xWS_RowNumber", "xWS_CartProd", "xWS_Nulls"
    ws_helper: pl.DataFrame = pl.DataFrame({
        ws_cart_prod: [
            xcp + 1 for xcp in going_from_roots_down_to_the_leaves
            for _ in range(xcp, - 1, - 1)
        ],
        ws_nulls: [
            len(ptr_key_cols) - xnull for xcp in going_from_roots_down_to_the_leaves
            for xnull in range(xcp, - 1, - 1)
        ]
    })

    ingest: pl.DataFrame = (
        ingest
        .lazy()

        .with_columns(
            pl.int_range(pl.len(), dtype=pl.UInt32).set_sorted()
            .alias(ws_row_number)
        )
        .with_columns(
            pl.col(ws_row_number).first().over(ptr_key_cols[:gdown + 1])
            .alias(f"xWS{gdown:03}")
            for gdown in going_from_roots_down_to_the_leaves[:-1]
        )
        .sort([f"xWS{gdown:03}" for gdown in going_from_roots_down_to_the_leaves[:-1]] + [ws_row_number])
        # ↑↑↑ step1 co-locate (cluster, arrange) the nodes (rows, paths)

        .with_columns(
            pl.sum_horizontal(
                pl.col(ws_row_number).eq(pl.col(f"xWS{gdown:03}"))
                for gdown in going_from_roots_down_to_the_leaves[:-1]
            )
            .add(1).alias(ws_cart_prod)
        )
        .join(ws_helper.lazy(), on=ws_cart_prod, how="inner", maintain_order="left")
        .with_columns(
            pl.when(pl.col(ws_nulls).gt(gdown))
            .then(pl.col(ptr_key_cols[gdown]))
            .otherwise(None)
            .alias(ptr_key_cols[gdown])
            for gdown in going_from_roots_down_to_the_leaves
        )
        # ↑↑↑ step2 explode the nodes (rows, paths)

        .select(~cs.starts_with("xWS"))

        .unique(keep="first", maintain_order=True, subset=ptr_key_cols)
        # ↑↑↑ step3 dedupe the nodes (rows, paths) - first arrival is kept
    )

    return ingest


def expand_model(ingest: pl.DataFrame | pl.LazyFrame, /, *, vis_all: bool = False, verify: int | None = None) -> pl.LazyFrame:
    """
    Computes integer-based boundaries and other metrics for each node in a tree DataFrame.
    In addition to assigning left and right pointers for nested set style queries, this function
    calculates values like depth, out-degree, and subtree size. These columns enable efficient
    subtree extraction in both Python (through slicing) and SQL (using BETWEEN clauses).
    By enriching the DataFrame with these bounds and metrics, downstream operations on hierarchical
    data become more flexible, performant, and intuitive.
    """
    # previously called build_trees
    # previously called from_mvp_flat_to_full
    # previously called to_full_from_mvp_flat

    going_from_roots_down_to_the_leaves: range = range(
        len(cs.expand_selector(ingest, cs_ptr)) if cs.expand_selector(ingest, cs_ptr)
        # from FLAT to FULL model
        else (ingest.lazy().select(pl.col(cdepth)).max().collect().to_series().item(0) + 1)
        # from MVP to FLAT model
    )
    going_from_leaves_up_to_the_roots: range = range(max(going_from_roots_down_to_the_leaves), - 1, - 1)
    # going from roots down(↓) to the leaves [0, 1, 2, 3, ...] ascending(↑) integers (cdepth)
    # going from leaves up(↑) to the roots [..., 3, 2, 1, 0] descending(↓) integers (cdepth)

    # ↓↓↓ from MVP to FLAT model
    if not cs.expand_selector(ingest, cs_ptr):
        ingest: pl.LazyFrame = (
            ingest.lazy()
            .with_columns(
                pl.when(pl.col(cdepth).lt(descent)).then(None)
                .otherwise(
                    pl.when(pl.col(cdepth).ne(descent)).then(None)
                    .otherwise(ctail).fill_null(strategy="forward")
                ).alias(f"p{descent:03}_{max(going_from_roots_down_to_the_leaves)}")
                for descent in going_from_roots_down_to_the_leaves
            )
        )
    # ↑↑↑ from MVP to FLAT model

    # ↓↓↓ from FLAT to FULL model
    ptr_key_cols: tuple[str, ...] = tuple(sorted(cs.expand_selector(ingest, cs_ptr)))
    assert ptr_key_cols and all(ptr_key_cols) and ptr_key_cols == tuple(cs.expand_selector(ingest, cs_ptr))
    assert (not verify) or ingest.lazy().select(cs_ptr).head(verify).collect().height == (
        ingest.lazy().select(cs_ptr).head(verify)
        .filter(
            ((cs_ptr & cs.first()).shift(1).is_null() & pl.sum_horizontal(cs_ptr.is_not_null()).eq(1)) |
            pl.concat_list(cs_ptr).list.head(pl.sum_horizontal(cs_ptr.is_not_null()).sub(1))
            .eq(pl.concat_list(cs_ptr).shift(1).list.head(pl.sum_horizontal(cs_ptr.is_not_null()).sub(1)))
        )
        .unique(keep="any", maintain_order=False)
        .collect()
        .height
    ), "The verify parameter is set, but the DataFrame contains identical rows or rows not colocated correctly"

    #
    # Best Measured Performance with default parameters is ≈ 11.3M rows/sec. The system processes 100,000 rows
    # in ≈ 8.85 ms, which is only 1.77% of the original 500 ms performance budget. This indicates exceptional
    # efficiency, with 98.23% of the target window unused, leaving ample headroom for additional workload.
    #

    for ascent in going_from_leaves_up_to_the_roots:
        if ascent == max(going_from_leaves_up_to_the_roots):
            build_working_storage = ()
            is_aligned: pl.Expr = pl.col(ptr_key_cols[ascent]).is_not_null()
            build_ctail: pl.Expr = pl.when(is_aligned).then(ptr_key_cols[ascent])
            build_cdepth: pl.Expr = pl.when(is_aligned).then(pl.lit(ascent, dtype=_sys_dtype))
            build_cstop: pl.Expr = pl.when(is_aligned).then(pl.lit(1))
            build_cwidth: pl.Expr = pl.when(is_aligned).then(pl.lit(True).filter(is_aligned).len())
            build_cout: pl.Expr = pl.when(is_aligned).then(0)
            continue
        build_working_storage += (pl.struct(pl.col(ptr_key_cols[ascent])).rle_id().set_sorted().alias(f"xWS{ascent:03}"),)
        is_aligned: pl.Expr = pl.col(ptr_key_cols[ascent]).is_not_null() & pl.col(ptr_key_cols[ascent + 1]).is_null()
        build_ctail: pl.Expr = build_ctail.when(is_aligned).then(ptr_key_cols[ascent])
        build_cdepth: pl.Expr = build_cdepth.when(is_aligned).then(pl.lit(ascent, dtype=_sys_dtype))
        build_cstop: pl.Expr = build_cstop.when(is_aligned).then(pl.col(f"xWS{ascent:03}").len().over(f"xWS{ascent:03}"))
        # build_cstop: pl.Expr = build_cstop.when(is_aligned).then(
        #     # correct result, looks compelling, but only for super SHALLOW WIDE TREES
        #     # +7.81% cpu: the cost of the test far outweighs the benefit
        #     pl.when(pl.col(ptr_key_cols[ascent + 1]).shift(-1).is_null()).then(
        #         pl.lit(1)
        #     ).otherwise(
        #         pl.col(f"xWS{ascent:03}").len().over(f"xWS{ascent:03}")
        #     )
        # )
        build_cwidth: pl.Expr = build_cwidth.when(is_aligned).then(pl.lit(True).filter(is_aligned).len())
        build_cout: pl.Expr = build_cout.when(is_aligned).then(
            (
                pl.col(f"xWS{ascent:03}")
                if ascent == max(going_from_leaves_up_to_the_roots) - 1 else
                # # correct result, looks compelling, but defintely NOT
                # # +200.00% cpu
                # pl.col(f"xWS{ascent + 1:03}").unique()
                pl.col(f"xWS{ascent:03}").filter(pl.col(ptr_key_cols[ascent + 2]).is_null())
            ).len().sub(1).over(f"xWS{ascent:03}")
        )

    build_cvis_a: tuple[pl.Expr, ...] = tuple(
        pl.when(pl.col(cdepth).eq(ascent)).then(pl.col(cstop))
        .fill_null(strategy="forward")
        .alias(f"v{ascent:03}_{max(going_from_leaves_up_to_the_roots)}")
        for ascent in going_from_leaves_up_to_the_roots
    )

    vis_all = pl.col(ctail).cast(dtype=pl.Utf8) if vis_all else pl.col(ctail).cast(dtype=pl.Utf8).str.strip_chars().str.slice(offset=0, length=9)

    build_cvis_b: tuple[pl.Expr, ...] = tuple((
        pl
        .when(pl.col(cdepth).lt(ascent)).then(vis_void)
        .when(pl.col(cdepth).ge(ascent + 2)).then(
            pl.when(pl.col(f"v{ascent:03}_{max(going_from_leaves_up_to_the_roots)}").eq(pl.col(f"v{ascent + 1:03}_{max(going_from_leaves_up_to_the_roots)}")))
            .then(vis_void).otherwise(vis_pipe)
        )
        .when(pl.col(cdepth).eq(ascent + 1)).then(
            pl.when(pl.col(f"v{ascent:03}_{max(going_from_leaves_up_to_the_roots)}").eq(pl.col(f"v{ascent + 1:03}_{max(going_from_leaves_up_to_the_roots)}")))
            .then(vis_closed).otherwise(vis_open)
        )
        .otherwise(vis_all)
        if ascent < max(going_from_leaves_up_to_the_roots) - 1 else

        pl
        .when(pl.col(cdepth).lt(ascent)).then(vis_void)
        .when(pl.col(cdepth).eq(ascent + 1)).then(
            pl.when(pl.col(f"v{ascent:03}_{max(going_from_leaves_up_to_the_roots)}").eq(pl.col(f"v{ascent + 1:03}_{max(going_from_leaves_up_to_the_roots)}")))
            .then(vis_closed).otherwise(vis_open)
        )
        .otherwise(vis_all)
        if ascent < max(going_from_leaves_up_to_the_roots) else

        pl
        .when(pl.col(cdepth).lt(ascent)).then(vis_void)
        .otherwise(vis_all)

    ).alias(f"v{ascent:03}_{max(going_from_leaves_up_to_the_roots)}") for ascent in going_from_leaves_up_to_the_roots)

    ingest: pl.LazyFrame = (
        ingest
        .lazy()
        .with_columns(
            pl.col(ptr_key_cols[0]).alias(chead),
            build_ctail.alias(ctail),
            build_cdepth.alias(cdepth),
            pl.int_range(pl.len(), dtype=_sys_dtype).set_sorted().alias(cstart),
            *build_working_storage
        )
        .with_columns(
            build_cstop.add(pl.col(cstart)).alias(cstop),
            # +88.72% combined cpu cost ↓↓↓
            build_cwidth.alias(cwidth),
            build_cout.alias(cout)
            # +88.72% combined cpu cost ↑↑↑
        )
        .with_columns(
            # +3.86% combined cpu cost ↓↓↓
            pl.col(cstop).sub(cstart).alias(csize),
            pl.col(cdepth).eq(0).alias(cis_root),
            pl.col(cstop).sub(cstart).eq(1).alias(cis_leaf),
            pl.col(cstart).mul(2).sub(cdepth).add(1).cast(dtype=_sys_dtype).alias(cleft),
            pl.col(cstop).mul(2).sub(cdepth).cast(dtype=_sys_dtype).alias(cright),
            # +3.86% combined cpu cost ↑↑↑
            *build_cvis_a[::-1]
        )
        .with_columns(*build_cvis_b)
        .select(
            cs_ptr, cs_vis, *_core_12, cs_hash, cs_uqu,
            ~(cs_ptr | cs_vis | cs_core_12 | cs_hash | cs_uqu | cs.starts_with("xWS"))
        )
    )
    # ↑↑↑ from FLAT to FULL model

    return ingest


def from_mList(ingest: pl.DataFrame, /) -> pl.DataFrame:

    assert tree_hook(ingest)["mList"][-1]

    # the first column is of type pl.List (the inner type is immaterial)
    assert (
        ingest
        .lazy()
        .select(
            pl.col(cptr).is_null().alias("is_null"),
            pl.col(cptr).list.len().alias("len"),
            pl.col(cptr).list.drop_nulls().list.len().alias("no_nulls")
        )
        .filter(pl.col("is_null") | pl.col("len").eq(0) | pl.col("len").ne(pl.col("no_nulls")))
        .head(1)
        .collect()
        .height == 0
    ), "The list column cannot be null, contain any nulls, or be zero length"

    ingest: pl.DataFrame = (
        ingest
        .drop(cs_ptr)
        .with_columns(
            pl.col(cptr).list.to_struct(
                n_field_strategy="max_width",
                fields=lambda depth: f"xWS{depth:03}"
            ).alias("yWS")
        )
        .unnest(columns="yWS")
        .drop(cptr)
    )

    going_from_leaves_up_to_the_roots: range = range(len(cs.expand_selector(ingest, cs.starts_with("xWS"))) - 1, - 1, - 1)
    # going from leaves up(↑) to the roots [..., 3, 2, 1, 0] descending(↓) integers (cdepth)
    return ingest.rename(
        {f"xWS{ascent:03}": f"p{ascent:03}_{max(going_from_leaves_up_to_the_roots)}" for ascent in going_from_leaves_up_to_the_roots}
    ).rechunk()


def from_mALPC(ingest: pl.DataFrame, /, *, max_depth: int = 100) -> pl.LazyFrame:
    # from ALIST to FULL model
    # previously called to_flat_from_alist

    assert tree_hook(ingest)["mALPC"][-1]

    user_val = cs.expand_selector(ingest, ~cs.by_name(cparent, cchild, cbkey, require_all=False))

    check_parent: pl.Series = ingest.get_column(cparent)
    assert ingest.height and ingest.height == ingest.get_column(cchild).unique().count()
    assert ingest.filter(pl.col(cparent).eq(pl.col(cchild))).height == 0
    assert check_parent.filter(check_parent.is_null()).shape[0] > 0
    assert check_parent.filter(check_parent.is_not_null() & check_parent.is_duplicated()).shape[0] > 0

    leaves_expr: pl.Expr = pl.col(cchild).is_in(
        check_parent.drop_nulls().unique().sort().implode().set_sorted(),
        nulls_equal=False
    ).not_()
    n_leaves, leaves = pl.collect_all((
        ingest.lazy().filter(~leaves_expr).select(cchild, cparent),
        ingest.lazy().filter(leaves_expr).select(cchild, cparent)
    ))
    del check_parent, leaves_expr

    while leaves.select(cs.last().is_not_null().any()).to_series().item(0) and leaves.shape[-1] < max_depth:
        leaves: pl.DataFrame = (
            leaves.lazy()
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cs.expand_selector(leaves, cs.last())[-1],    right_on=cchild, suffix=f"_{leaves.shape[-1]+0:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+0:03}join",    right_on=cchild, suffix=f"_{leaves.shape[-1]+1:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+1:03}join",    right_on=cchild, suffix=f"_{leaves.shape[-1]+2:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+2:03}join",    right_on=cchild, suffix=f"_{leaves.shape[-1]+3:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+3:03}join",    right_on=cchild, suffix=f"_{leaves.shape[-1]+4:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+4:03}join",    right_on=cchild, suffix=f"_{leaves.shape[-1]+5:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+5:03}join",    right_on=cchild, suffix=f"_{leaves.shape[-1]+6:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+6:03}join",    right_on=cchild, suffix=f"_{leaves.shape[-1]+7:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+7:03}join",    right_on=cchild, suffix=f"_{leaves.shape[-1]+8:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+8:03}join",    right_on=cchild, suffix=f"_{leaves.shape[-1]+9:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+9:03}join",    right_on=cchild, suffix=f"_{leaves.shape[-1]+10:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+10:03}join",   right_on=cchild, suffix=f"_{leaves.shape[-1]+11:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+11:03}join",   right_on=cchild, suffix=f"_{leaves.shape[-1]+12:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+12:03}join",   right_on=cchild, suffix=f"_{leaves.shape[-1]+13:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+13:03}join",   right_on=cchild, suffix=f"_{leaves.shape[-1]+14:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cparent + f"_{leaves.shape[-1]+14:03}join",   right_on=cchild, suffix=f"_{leaves.shape[-1]+15:03}join")
            .select(cchild, cs.starts_with(cparent))
            .collect()
        )

    del n_leaves

    leaves: pl.DataFrame = (
        leaves
        .lazy()
        .select(leaves.columns[:max_depth+1][:-(
            leaves
            .select(pl.sum_horizontal(pl.col(leaves.columns[:max_depth+1][-16:]).is_not_null().any().not_()))
            .to_series().item(0)
        )])
        .collect()
    )

    going_from_roots_down_to_the_leaves: range = range(len(leaves.columns))
    going_from_leaves_up_to_the_roots: range = range(max(going_from_roots_down_to_the_leaves), - 1, - 1)
    # going from roots down(↓) to the leaves [0, 1, 2, 3, ...] ascending(↑) integers (cdepth)
    # going from leaves up(↑) to the roots [..., 3, 2, 1, 0] descending(↓) integers (cdepth)

    ingest: pl.LazyFrame = (
        leaves
        .lazy()
        .select(
            pl.concat_list(pl.all()).list.reverse().list.drop_nulls()
            .list.to_struct(fields=
                [f"p{qq:03}_{max(going_from_roots_down_to_the_leaves)}" for qq in going_from_roots_down_to_the_leaves],
            ).struct.unnest()
        )
        .pipe(refine_model)
        .with_columns(pl.coalesce(pl.col([f"p{qq:03}_{max(going_from_leaves_up_to_the_roots)}" for qq in going_from_leaves_up_to_the_roots])).alias(uv) for uv in user_val[0:1])
        .with_columns(pl.col(user_val[0]).alias(uv) for uv in user_val)
        .with_columns(
            [cs_ptr.replace_strict(old=ingest.get_column(cchild), new=ingest.get_column(cbkey), default=None)] +
            [pl.col(uv).replace_strict(old=ingest.get_column(cchild), new=ingest.get_column(uv), default=None).alias(uv) for uv in user_val]
        )
    )

    del leaves

    return ingest


def test_equal(tree_x: pl.DataFrame, tree_y: pl.DataFrame, /) -> bool:
    tree_x = tree_x.select(chead, ctail, cdepth, cstart, cstop)
    tree_y = tree_y.select(tree_x.columns)
    if tree_x.equals(tree_y):
        return True
    else:
        if tree_x.select(cdepth, chead, ctail).sort(by=pl.all()).equals(tree_y.select(cdepth, chead, ctail).sort(by=pl.all())):
            tree_z: pl.DataFrame = pl.concat(how="horizontal", items=[

                tree_y.select(cstart, cstop)
                +  # proof that (dataframe level) element-wise matrix addition / subtraction works
                tree_y
                .join(how="inner", on=chead, suffix="_Δ", other=tree_x.filter(pl.col(cdepth).eq(0)).select(chead, cstart))
                .select(pl.col(cstart+"_Δ").cast(dtype=pl.Int32).sub(pl.col(cstart).min().over(chead)))
                .select(pl.col(cstart+"_Δ").alias("col1"), pl.col(cstart+"_Δ").alias("col2")),

                tree_y.drop(cstart, cstop)

            ]).select(tree_x.columns).with_columns(cs.numeric().cast(dtype=_sys_dtype)).sort(cstart)

            return tree_x.equals(tree_z)

    return False


if __name__ == "__main__":
    pl.show_versions()

    from data_design import example_mvp, example_alist, unit_test_aa_mvp, unit_test_bb_mvp, unit_test_cc_mvp

    import duckdb as db

    # demonstration

    assert pl.read_parquet(example_mvp).pipe(expand_model).collect().shape == (20, 22)
    assert pl.read_parquet(unit_test_aa_mvp).pipe(expand_model).collect().shape == (52_529, 32)
    assert pl.read_parquet(unit_test_bb_mvp).pipe(expand_model).collect().shape == (52_419, 32)
    assert pl.read_parquet(unit_test_cc_mvp).pipe(expand_model).collect().shape == (49_246, 32)

    p_sep = 'parent> <child'
    for this_df, this_node in [
        [example_mvp, "E"],
        [unit_test_cc_mvp, {"type": 99, "obj": "kX"}]
    ]:
        print(f"\n\n\nSQL Queries for: {this_df=}  {this_node=}  {type(this_node)=}", end="")
        this_df: pl.DataFrame = pl.read_parquet(this_df).select(~(cs.starts_with("user_value") | cs.by_name("BLOCK_ID"))).head(50).pipe(expand_model).collect()
        print(f"  {this_df.schema[ctail]=}")

        query0 = f"""SELECT * FROM this_df;"""
        assert db.sql(query0).pl().equals(
            this_df
        )
        # show base data; test against equivalent Polars DF
        db.sql(query0).show(max_rows=50, max_width=5_000)

        query1 = f"""SELECT * FROM this_df WHERE {ctail}={repr(this_node)};"""
        assert db.sql(query1).pl().equals(
            this_df.lazy().filter(pl.col(ctail).eq(this_node)).collect()
        )
        # get all; test against equivalent Polars DF
        db.sql(query1).show(max_rows=50, max_width=5_000)

        query2 = f"""
        SELECT P.{cleft} AS P_{cleft}, P.{cright} AS P_{cright}, {repr(p_sep)} AS 'P_SEP', c.*
        FROM this_df AS P, this_df AS C
        WHERE c.{cleft} BETWEEN P.{cleft} AND P.{cright} AND P.{ctail}={repr(this_node)}
        ORDER BY P.{cleft} ASC, c.{cleft} ASC;
        """
        assert db.sql(query2).pl().equals(
            this_df.lazy()
            .filter(pl.col(ctail).eq(this_node))
            .select(pl.col(cleft, cright).name.prefix("P_"), pl.lit(p_sep).alias("P_SEP"))
            .set_sorted("P_" + cleft)
            .join_where(
                this_df.lazy().set_sorted(cleft),
                pl.col(cleft) >= pl.col("P_" + cleft),
                pl.col(cleft) < pl.col("P_" + cright)
            )
            .sort("P_" + cleft, cleft)
            .collect()
        )
        assert db.sql(query2).pl().equals(
            pl.concat(items=(
                this_df.slice(offset=Δ, length=囗)
                .select(
                    pl.when(pl.col(cstart).eq(Δ)).then(pl.col(cleft, cright)).otherwise(None)
                    .fill_null(strategy="forward").name.prefix("P_"),
                    pl.lit(p_sep).alias("P_SEP"),
                    pl.all()
                )
                for Δ, 囗 in this_df.filter(pl.col(ctail).eq(this_node)).select([cstart, csize]).iter_rows()
            ), how="vertical")
            # .rechunk()
        )
        # get full subtrees using cleft and cright (for all); test against equivalent Polars DF
        db.sql(query2).show(max_rows=50, max_width=5_000)


    this_df: pl.DataFrame = (
        pl.read_parquet(example_alist)
        .rename({"parent": cparent, "child": cchild, "business_key": cbkey})
    )
    # show an adjacency list model
    db.sql(f"""SELECT * FROM this_df;""").show(max_rows=50, max_width=5_000)
    this_df: pl.DataFrame = (
        this_df
        .pipe(from_mALPC)
        .pipe(expand_model).collect()
    )
    # demo pipeline to treat an adjacency list model
    db.sql(f"""SELECT * FROM this_df;""").show(max_rows=50, max_width=5_000)
