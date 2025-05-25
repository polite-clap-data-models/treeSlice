import polars as pl
import polars.selectors as cs
import polars_hash as plh

chead, ctail, cdepth, cstart, cstop = "cHead", "cTail", "cDepth", "cStart", "cStop"
hk_ptr, hk_ptr_parent, hk_schema, hk_nspace, hk_kvp_treekey = "hkPTR", "hkPTR_alParent", "hkSchema", "hkNSpace", "hkKVP_TreeKey"
csize, cwidth, cout = "cSize", "cWidth", "cOut"
cleft, cright, cis_root, cis_leaf = "cLeft", "cRight", "cIsRoot", "cIsLeaf"
vis_open, vis_closed, vis_pipe, vis_void = (
    pl.lit(value=" ╠══➤ ", dtype=pl.String), pl.lit(value=" ╚══➤ ", dtype=pl.String),
    pl.lit(value=" ║   ", dtype=pl.String), pl.lit(value="  ", dtype=pl.String)
)

_sys_dtype = pl.DataFrame(schema={cstart: pl.UInt32}).schema[cstart]
cs_ptr = cs.matches(pattern="^p\\d{3}_[1-9]\\d*$")
cs_vis = cs.matches(pattern="^v\\d{3}_[1-9]\\d*$")
cs_mini = cs.by_name(chead, ctail, cdepth, cstart, cstop, require_all=True)
cs_hash = cs.by_name(hk_ptr, hk_ptr_parent, hk_schema, hk_nspace, hk_kvp_treekey, require_all=False)
cs_from = cs.by_name(csize, cwidth, cout, cleft, cright, cis_root, cis_leaf, require_all=False)
cs_uqu = (
    cs.starts_with("uq") &
    cs.by_dtype(pl.Struct({"cIsLCA": pl.Boolean, "cIsUQE": pl.Boolean, "cPTR": pl.List(_sys_dtype)}))
)


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


def flat_unpack_list(ingest: pl.DataFrame, /) -> pl.DataFrame:

    first_column: str = cs.expand_selector(ingest, cs.first())[0]

    if repr(ingest.schema[first_column])[:5] != "List(":
        return ingest

    # the first column is of type pl.List (the inner type is immaterial)
    assert (
        ingest
        .lazy()
        .select(
            pl.col(first_column).is_null().alias("is_null"),
            pl.col(first_column).list.len().alias("len"),
            pl.col(first_column).list.drop_nulls().list.len().alias("no_nulls")
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
            pl.col(first_column).list.to_struct(
                n_field_strategy="max_width",
                fields=lambda depth: f"xWS{depth:03}"
            ).alias("yWS")
        )
        .unnest(columns="yWS")
        .drop(first_column)
    )

    going_from_leaves_up_to_the_roots: range = range(len(cs.expand_selector(ingest, cs.starts_with("xWS"))) - 1, - 1, - 1)
    # going from leaves up(↑) to the roots [..., 3, 2, 1, 0] descending(↓) integers (cdepth)
    return ingest.rename(
        {f"xWS{ascent:03}": f"p{ascent:03}_{max(going_from_leaves_up_to_the_roots)}" for ascent in going_from_leaves_up_to_the_roots}
    ).rechunk()


def flat_reorg_ced(ingest: pl.DataFrame | pl.LazyFrame, /) -> pl.LazyFrame:
    """
    Refines a DataFrame of tree paths by ensuring completeness and proper ordering.
    It deduplicates rows, co-locates sibling nodes under their shared parent, and
    generates any missing intermediate nodes for leaf-only paths. The result is a
    coherent, fully expanded hierarchy that accurately represents each node's position
    in the tree. Ideal for downstream processing or visualisation of hierarchical data.
    """
    # flat_reorg_ced (co-locate, explode, dedupe)

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


def to_full_from_mvp_flat(ingest: pl.DataFrame | pl.LazyFrame, /, *, vis_shorten: bool = True) -> pl.LazyFrame:
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

    vis_shorten = pl.col(ctail).cast(dtype=pl.Utf8).str.strip_chars().str.slice(offset=0, length=9) if vis_shorten else pl.col(ctail).cast(dtype=pl.Utf8).str.strip_chars()

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
        .otherwise(vis_shorten)
        if ascent < max(going_from_leaves_up_to_the_roots) - 1 else

        pl
        .when(pl.col(cdepth).lt(ascent)).then(vis_void)
        .when(pl.col(cdepth).eq(ascent + 1)).then(
            pl.when(pl.col(f"v{ascent:03}_{max(going_from_leaves_up_to_the_roots)}").eq(pl.col(f"v{ascent + 1:03}_{max(going_from_leaves_up_to_the_roots)}")))
            .then(vis_closed).otherwise(vis_open)
        )
        .otherwise(vis_shorten)
        if ascent < max(going_from_leaves_up_to_the_roots) else

        pl
        .when(pl.col(cdepth).lt(ascent)).then(vis_void)
        .otherwise(vis_shorten)

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
        .select(chead, ctail, cdepth, cstart, cstop, ~cs_mini)
        .select(
            cs_ptr, cs_vis, cs_mini, cs_from, cs_hash, cs_uqu,
            ~(cs_ptr | cs_vis | cs_mini | cs_from | cs_hash | cs_uqu | cs.starts_with("xWS"))
        )
    )
    # ↑↑↑ from FLAT to FULL model

    return ingest


def to_flat_from_alist(ingest: pl.DataFrame, /, *, max_depth: int = 100) -> pl.LazyFrame:
    """
    it is the ORDINAL POSITION of the values which is material (NOT the column names used)
    0, 1: parent and child
    2: the business key
    *3: an arbitrary number of business values
    """

    alist_parent_sk, alist_child_sk, user_key, *user_val = ingest.columns
    ingest: pl.DataFrame = ingest.rename({alist_parent_sk: alist_parent_sk + "_xWS", alist_child_sk: alist_child_sk + "_xWS", user_key: user_key + "_xWS"})
    alist_parent_sk, alist_child_sk, user_key, *user_val = ingest.columns

    check_parent: pl.Series = ingest.get_column(alist_parent_sk)
    assert ingest.height and ingest.height == ingest.get_column(alist_child_sk).unique().count()
    assert ingest.filter(pl.col(alist_parent_sk).eq(pl.col(alist_child_sk))).height == 0
    assert check_parent.filter(check_parent.is_null()).shape[0] > 0
    assert check_parent.filter(check_parent.is_not_null() & check_parent.is_duplicated()).shape[0] > 0

    leaves_expr: pl.Expr = pl.col(alist_child_sk).is_in(
        check_parent.drop_nulls().unique().sort().implode().set_sorted(),
        nulls_equal=False
    ).not_()
    n_leaves, leaves = pl.collect_all((
        ingest.lazy().filter(~leaves_expr).select(alist_child_sk, alist_parent_sk),
        ingest.lazy().filter(leaves_expr).select(alist_child_sk, alist_parent_sk)
    ))
    del check_parent, leaves_expr

    while leaves.select(cs.last().is_not_null().any()).to_series().item(0) and leaves.shape[-1] < max_depth:
        leaves: pl.DataFrame = (
            leaves.lazy()
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=cs.expand_selector(leaves, cs.last())[-1],            right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+0:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+0:03}join",    right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+1:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+1:03}join",    right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+2:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+2:03}join",    right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+3:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+3:03}join",    right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+4:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+4:03}join",    right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+5:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+5:03}join",    right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+6:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+6:03}join",    right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+7:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+7:03}join",    right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+8:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+8:03}join",    right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+9:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+9:03}join",    right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+10:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+10:03}join",   right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+11:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+11:03}join",   right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+12:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+12:03}join",   right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+13:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+13:03}join",   right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+14:03}join")
            .join(other=n_leaves.lazy(), how="left", nulls_equal=False, left_on=alist_parent_sk + f"_{leaves.shape[-1]+14:03}join",   right_on=alist_child_sk, suffix=f"_{leaves.shape[-1]+15:03}join")
            .select(alist_child_sk, cs.starts_with(alist_parent_sk))
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
        .pipe(flat_reorg_ced)
        .with_columns(pl.coalesce(pl.col([f"p{qq:03}_{max(going_from_leaves_up_to_the_roots)}" for qq in going_from_leaves_up_to_the_roots])).alias(uv) for uv in user_val[0:1])
        .with_columns(pl.col(user_val[0]).alias(uv) for uv in user_val)
        .with_columns(
            [cs_ptr.replace_strict(old=ingest.get_column(alist_child_sk), new=ingest.get_column(user_key), default=None)] +
            [pl.col(uv).replace_strict(old=ingest.get_column(alist_child_sk), new=ingest.get_column(uv), default=None).alias(uv) for uv in user_val]
        )
    )

    del leaves

    return ingest


def test_equal(tree_x: pl.DataFrame, tree_y: pl.DataFrame, /) -> bool:
    tree_x = tree_x.select(cs_mini)
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

    with pl.Config(
        thousands_separator="_", tbl_width_chars=1_200,
        fmt_str_lengths=80, fmt_table_cell_list_len=20,
        tbl_cols=200, tbl_rows=50
    ):

        # demonstration

        assert pl.read_parquet(example_mvp).pipe(to_full_from_mvp_flat).collect().shape == (20, 22)
        assert pl.read_parquet(unit_test_aa_mvp).pipe(to_full_from_mvp_flat).collect().shape == (52_529, 32)
        assert pl.read_parquet(unit_test_bb_mvp).pipe(to_full_from_mvp_flat).collect().shape == (52_419, 32)
        assert pl.read_parquet(unit_test_cc_mvp).pipe(to_full_from_mvp_flat).collect().shape == (49_246, 32)

        print(pl.read_parquet(example_mvp))
        print(
            pl.read_parquet(example_mvp)
            .pipe(to_full_from_mvp_flat).collect()
            .pipe(gen_hash_keys, as_uint64=False)
            .select(~cs_ptr)
        )

        print(pl.read_parquet(example_alist))
        assert pl.read_parquet(example_mvp).pipe(to_full_from_mvp_flat).collect().equals(
            pl.read_parquet(example_alist).pipe(to_flat_from_alist).pipe(to_full_from_mvp_flat).collect()
        )

        # demonstration - to_flat_from_alist

        unit_test_output: pl.DataFrame = (
            pl.scan_parquet(unit_test_cc_mvp)
            .pipe(to_full_from_mvp_flat)
            .with_columns(pl.col(cstart).alias("user_value1"), pl.col(cstop).alias("user_value2"))
            .collect().rechunk()
        )

        ptr_key_cols: tuple[str, ...] = tuple(sorted(cs.expand_selector(unit_test_output, cs_ptr)))

        unit_test_input: pl.DataFrame = (
            pl.concat(how="horizontal", items=[unit_test_output,
                unit_test_output.lazy()
                .select(
                    pl.when(pl.col(cdepth).ne(lvl)).then(pl.col(ptr_key_cols[lvl])).alias(ptr_key_cols[lvl])
                    for lvl in range(len(ptr_key_cols))
                )
                .join(
                    on=ptr_key_cols, how="left", nulls_equal=True,
                    other=unit_test_output.lazy().filter(~pl.col(cis_leaf))
                )
                .select(pl.col(cstart).alias("Parent"))
                .collect()
            ])
            .select(
                pl.col("Parent"), pl.col(cstart).alias("Child"),
                pl.col(ctail).alias("BKEY"), cs.starts_with("user_value")
            )
        )

        assert unit_test_output.select(cs_mini, "user_value1", "user_value2").equals(
            unit_test_input.pipe(to_flat_from_alist)
            .pipe(to_full_from_mvp_flat).select(cs_mini, "user_value1", "user_value2").collect()
        )
