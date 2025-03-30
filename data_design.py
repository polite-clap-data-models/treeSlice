import polars as pl

# the df literal representation of the xx series files above
unit_test_xx_r20_c24_h4_d4_string: pl.DataFrame = pl.DataFrame([
    pl.Series('p000_3', ['A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A2', 'A2', 'A2', 'A2', 'A2', 'A2', 'Z', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y'], dtype=pl.String),
    pl.Series('p001_3', [None, 'B', 'B', 'B', 'E', 'E', 'E', None, 'E', 'E', 'E', 'E', 'B', None, None, 'V', 'V', 'W', 'X', 'X'], dtype=pl.String),
    pl.Series('p002_3', [None, None, 'C', 'D', None, 'F', 'G', None, None, 'E', 'G', 'G', None, None, None, None, 'V', None, None, 'X'], dtype=pl.String),
    pl.Series('p003_3', [None, None, None, None, None, None, None, None, None, None, None, 'C', None, None, None, None, None, None, None, None], dtype=pl.String),
    pl.Series('cHead', ['A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A2', 'A2', 'A2', 'A2', 'A2', 'A2', 'Z', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y'], dtype=pl.String),
    pl.Series('cTail', ['A1', 'B', 'C', 'D', 'E', 'F', 'G', 'A2', 'E', 'E', 'G', 'C', 'B', 'Z', 'Y', 'V', 'V', 'W', 'X', 'X'], dtype=pl.String),
    pl.Series('cDepth', [0, 1, 2, 2, 1, 2, 2, 0, 1, 2, 2, 3, 1, 0, 0, 1, 2, 1, 1, 2], dtype=pl.UInt32),
    pl.Series('cStart', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=pl.UInt32),
    pl.Series('cStop', [7, 4, 3, 4, 7, 6, 7, 13, 12, 10, 12, 12, 13, 14, 20, 17, 17, 18, 20, 20], dtype=pl.UInt32),
    pl.Series('cWidth', [4, 7, 8, 8, 7, 8, 8, 4, 7, 8, 8, 1, 7, 4, 4, 7, 8, 7, 7, 8], dtype=pl.UInt32),
    pl.Series('cOut', [2, 2, 0, 0, 2, 0, 0, 2, 2, 0, 1, 0, 0, 0, 3, 1, 0, 0, 1, 0], dtype=pl.UInt32),
    pl.Series('cSize', [7, 3, 1, 1, 3, 1, 1, 6, 4, 1, 2, 1, 1, 1, 6, 2, 1, 1, 2, 1], dtype=pl.UInt32),
    pl.Series('cIsRoot', [True, False, False, False, False, False, False, True, False, False, False, False, False, True, True, False, False, False, False, False], dtype=pl.Boolean),
    pl.Series('cIsLeaf', [False, False, True, True, False, True, True, False, False, True, False, True, True, True, False, False, True, True, False, True], dtype=pl.Boolean),
    pl.Series('cLeft', [1, 2, 3, 5, 8, 9, 11, 15, 16, 17, 19, 20, 24, 27, 29, 30, 31, 34, 36, 37], dtype=pl.UInt32),
    pl.Series('cRight', [14, 7, 4, 6, 13, 10, 12, 26, 23, 18, 22, 21, 25, 28, 40, 33, 32, 35, 39, 38], dtype=pl.UInt32),
    pl.Series('v000_3', ['A1', ' ╠══➤ ', ' ║   ', ' ║   ', ' ╚══➤ ', '  ', '  ', 'A2', ' ╠══➤ ', ' ║   ', ' ║   ', ' ║   ', ' ╚══➤ ', 'Z', 'Y', ' ╠══➤ ', ' ║   ', ' ╠══➤ ', ' ╚══➤ ', '  '], dtype=pl.String),
    pl.Series('v001_3', ['  ', 'B', ' ╠══➤ ', ' ╚══➤ ', 'E', ' ╠══➤ ', ' ╚══➤ ', '  ', 'E', ' ╠══➤ ', ' ╚══➤ ', '  ', 'B', '  ', '  ', 'V', ' ╚══➤ ', 'W', 'X', ' ╚══➤ '], dtype=pl.String),
    pl.Series('v002_3', ['  ', '  ', 'C', 'D', '  ', 'F', 'G', '  ', '  ', 'E', 'G', ' ╚══➤ ', '  ', '  ', '  ', '  ', 'V', '  ', '  ', 'X'], dtype=pl.String),
    pl.Series('v003_3', ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ', 'C', '  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '], dtype=pl.String),
    pl.Series('hkPTR', ['484a_36038a14_a36f', '34f4_8518ef31_459c', '9c2a_76d5b5f7_81e1', '452b_748adedf_8874', '04a7_75ac6ab2_fcd0', 'f752_13e27807_98da', 'd950_f0af0e4b_19df', 'c7bd_635aea2b_03a8', 'fc89_55c2d279_4805', '03fe_9cecf8af_c48d', '438d_a9fd52fb_f3a6', '9db3_07fb99bc_169e', '21f0_315e7092_9dd4', 'b136_43cf6fec_ffac', 'ebdd_4ae19cd3_6355', 'fab3_1c6c0330_c54e', '6cc2_c75f2838_9793', '3807_1762c94f_67e9', 'd9cb_fe58a415_0c7d', '28c8_be6eb1ec_a0fe'], dtype=pl.String),
    pl.Series('hkSchema', ['4d31_fb8f36e8_7f12', '3b63_d94adf0f_98e3', 'e1ee_1e667642_cacf', 'e1ee_1e667642_cacf', '3b63_d94adf0f_98e3', 'e1ee_1e667642_cacf', 'e1ee_1e667642_cacf', '9160_7ab7c67d_7a46', '0d90_04f52e76_eb2a', 'e1ee_1e667642_cacf', 'f4ac_ee4a8f99_3f66', 'e1ee_1e667642_cacf', 'e1ee_1e667642_cacf', 'e1ee_1e667642_cacf', 'd7ff_7e893dc3_9ac9', 'f4ac_ee4a8f99_3f66', 'e1ee_1e667642_cacf', 'e1ee_1e667642_cacf', 'f4ac_ee4a8f99_3f66', 'e1ee_1e667642_cacf'], dtype=pl.String),
    pl.Series('hkNSpace', ['ceae_8725bb75_6fc6', '2cf1_db652151_6ca3', 'ce66_af02d168_85fc', 'ce66_af02d168_85fc', 'b782_1a5a02dc_6482', 'ce66_af02d168_85fc', 'ce66_af02d168_85fc', '3248_91d0fba2_f1a6', 'f27a_3f76c750_99f3', 'ce66_af02d168_85fc', '085f_5f113ad3_0600', 'ce66_af02d168_85fc', 'ce66_af02d168_85fc', 'ce66_af02d168_85fc', '1753_af49d5a0_bc18', '1f1a_f36f35bb_37cc', 'ce66_af02d168_85fc', 'ce66_af02d168_85fc', '5226_1aab0604_7533', 'ce66_af02d168_85fc'], dtype=pl.String),
    pl.Series('hkKVP_TreeKey', ['e6f7_732ad55c_0486', 'e6f7_732ad55c_0486', 'e6f7_732ad55c_0486', 'e6f7_732ad55c_0486', 'e6f7_732ad55c_0486', 'e6f7_732ad55c_0486', 'e6f7_732ad55c_0486', '9d96_c101f211_1f92', '9d96_c101f211_1f92', '9d96_c101f211_1f92', '9d96_c101f211_1f92', '9d96_c101f211_1f92', '9d96_c101f211_1f92', '497b_362d549a_e171', '746d_e25246ff_a6f9', '746d_e25246ff_a6f9', '746d_e25246ff_a6f9', '746d_e25246ff_a6f9', '746d_e25246ff_a6f9', '746d_e25246ff_a6f9'], dtype=pl.String),
])
unit_test_xx1: pl.DataFrame = pl.DataFrame([
    pl.Series('cPath', [['A1'], ['A1', 'B'], ['A1', 'B', 'C'], ['A1', 'B', 'D'], ['A1', 'E'], ['A1', 'E', 'F'], ['A1', 'E', 'G'], ['A2'], ['A2', 'E'], ['A2', 'E', 'E'], ['A2', 'E', 'G'], ['A2', 'E', 'G', 'C'], ['A2', 'B'], ['Z'], ['Y'], ['Y', 'V'], ['Y', 'V', 'V'], ['Y', 'W'], ['Y', 'X'], ['Y', 'X', 'X']], dtype=pl.List(pl.String)),
])
unit_test_xx2: pl.DataFrame = pl.DataFrame([
    pl.Series('cPath', [['A1', 'B', 'C'], ['A1', 'B', 'D'], ['A1', 'E', 'F'], ['A1', 'E', 'G'], ['A2', 'E', 'E'], ['A2', 'E', 'G', 'C'], ['A2', 'B'], ['Z'], ['Y', 'V', 'V'], ['Y', 'W'], ['Y', 'X', 'X']], dtype=pl.List(pl.String)),
])
unit_test_xx3: pl.DataFrame = pl.DataFrame([
    pl.Series('cPath', [['A1', 'B', 'C'], ['A1', 'E', 'F'], ['A2', 'E', 'E'], ['A2', 'E', 'G', 'C'], ['Z'], ['Y', 'V', 'V'], ['Y', 'W'], ['Y', 'X', 'X'], ['A1'], ['A1', 'B'], ['A1', 'B', 'D'], ['A1', 'E'], ['A1', 'E', 'G'], ['A2'], ['A2', 'E'], ['A2', 'E', 'G'], ['A2', 'B'], ['Y'], ['Y', 'V'], ['Y', 'X'], ['A1', 'B'], ['Y', 'V', 'V'], ['A2', 'B'], ['A1', 'E'], ['A2', 'E', 'E'], ['A2', 'E'], ['A1'], ['A2']], dtype=pl.List(pl.String)),
])



if __name__ == "__main__":

    from json import loads

    import polars.selectors as cs
    from polars.testing import assert_frame_equal

    from treeSlice import load_tree, gen_hash_keys, build_trees, unpack_cpath, refine_paths
    _tested_dtypes = (
        cs.string(include_categorical=True) |
        cs.numeric() |
        cs.by_dtype(pl.Enum) |
        cs.by_dtype(pl.Struct({"type": pl.Int64, "obj": pl.String})) |
        cs.by_dtype(pl.Array(inner=pl.String, shape=2))
    )

    df_list_of_Nx_int = pl.DataFrame({"list[i64]": [
        [9, 8, 4, 4], [9, 7], [9, 1, 5, 4], [9, 8, 3, 4], [9, 1, 5, 4], [1], [9, 7, 0, 0],
        [9, 1, 5, 5, 2], [9, 1, 5, 5, 5], [9, 1, 5, 5, 7], [9, 1, 5, 5, 9], [9, 1, 8]
    ]})
    df_list_of_Nx_str = pl.DataFrame({"list[str]": [
        ["9", "8", "4", "4"], ["9", "7"], ["9", "1", "5", "4"], ["9", "8", "3", "4"], ["9", "1", "5", "4"], ["1"], ["9", "7", "0", "0"],
        ["9", "1", "5", "5", "2"], ["9", "1", "5", "5", "5"], ["9", "1", "5", "5", "7"], ["9", "1", "5", "5", "9"], ["9", "1", "8"]
    ]})
    df_list_of_Nx_float = pl.DataFrame({"list[f64]": [
        [9.3, 8.3, 4.3, 4.3], [9.3, 7.3], [9.3, 1.3, 5.3, 4.3], [9.3, 8.3, 3.3, 4.3], [9.3, 1.3, 5.3, 4.3], [1.3], [9.3, 7.3, 0.3, 0.3],
        [9.3, 1.3, 5.3, 5.3, 2.3], [9.3, 1.3, 5.3, 5.3, 5.3], [9.3, 1.3, 5.3, 5.3, 7.3], [9.3, 1.3, 5.3, 5.3, 9.3], [9.3, 1.3, 8.3]
    ]})
    df_array_of_4xFixed_int = pl.DataFrame({"qqq": [
        [9, 8, 4, 4], [9, 1, 5, 4], [9, 8, 3, 4], [9, 1, 5, 4], [9, 7, 0, 0],
        [9, 1, 5, 5], [9, 1, 5, 5], [9, 1, 5, 5], [9, 1, 5, 5], [9, 1, 8, 0]
    ]}).select(pl.col("qqq").list.to_array(width=4).alias("array[i64, 4]"))
    df_JSON_array_of_4xFixed_int = pl.DataFrame({"str": [
        '[9, 8, 4, 4]', '[9, 1, 5, 4]', '[9, 8, 3, 4]', '[9, 1, 5, 4]', '[9, 7, 0, 0]',
        '[9, 1, 5, 5]', '[9, 1, 5, 5]', '[9, 1, 5, 5]', '[9, 1, 5, 5]', '[9, 1, 8, 0]'
    ]})
    assert all(len(loads(xxx)) == 4 for xxx in df_JSON_array_of_4xFixed_int.get_column("str").to_list())
    df_JSON_array_of_4xFixed_str = pl.DataFrame({"str": [
        '["9", "8", "4", "4"]', '["9", "1", "5", "4"]', '["9", "8", "3", "4"]', '["9", "1", "5", "4"]', '["9", "7", "0", "0"]',
        '["9", "1", "5", "5"]', '["9", "1", "5", "5"]', '["9", "1", "5", "5"]', '["9", "1", "5", "5"]', '["9", "1", "8", "0"]'
    ]})
    assert all(len(loads(xxx)) == 4 for xxx in df_JSON_array_of_4xFixed_str.get_column("str").to_list())
    df_JSON_array_of_Nx_JSON_object2 = pl.DataFrame({"str": [
        '[{"type": "x", "obj": 9}, {"type": "x", "obj": 8}, {"type": "x", "obj": 4}, {"type": "x", "obj": 4}]', '[{"type": "x", "obj": 9}, {"type": "x", "obj": 7}]', '[{"type": "x", "obj": 9}, {"type": "x", "obj": 1}, {"type": "x", "obj": 5}, {"type": "x", "obj": 4}]', '[{"type": "x", "obj": 9}, {"type": "x", "obj": 8}, {"type": "x", "obj": 3}, {"type": "x", "obj": 4}]', '[{"type": "x", "obj": 9}, {"type": "x", "obj": 1}, {"type": "x", "obj": 5}, {"type": "x", "obj": 4}]', '[{"type": "x", "obj": 1}]', '[{"type": "x", "obj": 9}, {"type": "x", "obj": 7}, {"type": "x", "obj": 0}, {"type": "x", "obj": 0}]',
        '[{"type": "y", "obj": 9}, {"type": "y", "obj": 1}, {"type": "y", "obj": 5}, {"type": "y", "obj": 5}, {"type": "y", "obj": 2}]', '[{"type": "y", "obj": 9}, {"type": "y", "obj": 1}, {"type": "y", "obj": 5}, {"type": "y", "obj": 5}, {"type": "y", "obj": 5}]', '[{"type": "y", "obj": 9}, {"type": "y", "obj": 1}, {"type": "y", "obj": 5}, {"type": "y", "obj": 5}, {"type": "y", "obj": 7}]', '[{"type": "y", "obj": 9}, {"type": "y", "obj": 1}, {"type": "y", "obj": 5}, {"type": "y", "obj": 5}, {"type": "y", "obj": 9}]', '[{"type": "y", "obj": 9}, {"type": "y", "obj": 1}, {"type": "y", "obj": 8}]'
    ]})
    assert all(all(len(xx) == 2 for xx in loads(xxx)) for xxx in df_JSON_array_of_Nx_JSON_object2.get_column("str").to_list())
    df_list_of_Nx_struct2 = pl.DataFrame({"list[struct[2]]": [
        [{"type": "x", "obj": 9}, {"type": "x", "obj": 8}, {"type": "x", "obj": 4}, {"type": "x", "obj": 4}], [{"type": "x", "obj": 9}, {"type": "x", "obj": 7}], [{"type": "x", "obj": 9}, {"type": "x", "obj": 1}, {"type": "x", "obj": 5}, {"type": "x", "obj": 4}], [{"type": "x", "obj": 9}, {"type": "x", "obj": 8}, {"type": "x", "obj": 3}, {"type": "x", "obj": 4}], [{"type": "x", "obj": 9}, {"type": "x", "obj": 1}, {"type": "x", "obj": 5}, {"type": "x", "obj": 4}], [{"type": "x", "obj": 1}], [{"type": "x", "obj": 9}, {"type": "x", "obj": 7}, {"type": "x", "obj": 0}, {"type": "x", "obj": 0}],
        [{"type": "y", "obj": 9}, {"type": "y", "obj": 1}, {"type": "y", "obj": 5}, {"type": "y", "obj": 5}, {"type": "y", "obj": 2}], [{"type": "y", "obj": 9}, {"type": "y", "obj": 1}, {"type": "y", "obj": 5}, {"type": "y", "obj": 5}, {"type": "y", "obj": 5}], [{"type": "y", "obj": 9}, {"type": "y", "obj": 1}, {"type": "y", "obj": 5}, {"type": "y", "obj": 5}, {"type": "y", "obj": 7}], [{"type": "y", "obj": 9}, {"type": "y", "obj": 1}, {"type": "y", "obj": 5}, {"type": "y", "obj": 5}, {"type": "y", "obj": 9}], [{"type": "y", "obj": 9}, {"type": "y", "obj": 1}, {"type": "y", "obj": 8}]
    ]})
    df_list_of_Nx_list2 = pl.DataFrame({"list[list[i64]]": [
        [[45, 9], [45, 8], [45, 4], [45, 4]], [[45, 9], [45, 7]], [[45, 9], [45, 1], [45, 5], [45, 4]], [[45, 9], [45, 8], [45, 3], [45, 4]], [[45, 9], [45, 1], [45, 5], [45, 4]], [[45, 1]], [[45, 9], [45, 7], [45, 0], [45, 0]],
        [[46, 9], [46, 1], [46, 5], [46, 5], [46, 2]], [[46, 9], [46, 1], [46, 5], [46, 5], [46, 5]], [[46, 9], [46, 1], [46, 5], [46, 5], [46, 7]], [[46, 9], [46, 1], [46, 5], [46, 5], [46, 9]], [[46, 9], [46, 1], [46, 8]]
    ]})
    df_list_of_Nx_array2 = df_list_of_Nx_list2.select(pl.col("list[list[i64]]").list.eval(pl.element().list.to_array(width=2)).alias("list[array[i64, 2]]"))


    with pl.Config(
        thousands_separator="_", tbl_width_chars=1_200,
        fmt_str_lengths=80, fmt_table_cell_list_len=20,
        tbl_cols=200, tbl_rows=50
    ):

        assert_frame_equal(
            load_tree(unit_test_xx_r20_c24_h4_d4_string),
            load_tree(unit_test_xx1).pipe(unpack_cpath)#.pipe(refine_paths)
            .pipe(build_trees).collect().pipe(gen_hash_keys),
            check_exact=True, check_row_order=True, check_column_order=True
        )
        assert_frame_equal(
            load_tree(unit_test_xx_r20_c24_h4_d4_string),
            load_tree(unit_test_xx2).pipe(unpack_cpath).pipe(refine_paths)
            .pipe(build_trees).collect().pipe(gen_hash_keys),
            check_exact=True, check_row_order=True, check_column_order=True
        )
        assert_frame_equal(
            load_tree(unit_test_xx_r20_c24_h4_d4_string),
            load_tree(unit_test_xx3).pipe(unpack_cpath).pipe(refine_paths)
            .pipe(build_trees).collect().pipe(gen_hash_keys),
            check_exact=True, check_row_order=True, check_column_order=True
        )

    print("NO REGRESSIONS detected")
