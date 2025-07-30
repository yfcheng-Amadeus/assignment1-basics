import cProfile
import pstats
import io
from BPETokenizer import BPETokenizer # 假设你的类在 bpe_tokenizer.py 文件中

if __name__ == "__main__":
    tokenizer = BPETokenizer()
    input_file = "/Users/yfcheng/Desktop/cs336/assignment1-basics/tests/fixtures/corpus.en"
    vocab_size=1000
    special_tokens=["<|endoftext|>"]

    # 使用 cProfile.Profile() 对象进行分析
    pr = cProfile.Profile()

    print(f"Starting BPE training profiling on {input_file} with vocab_size={vocab_size}...")

    pr.enable() # 开始收集性能数据

    # 调用你想要分析的方法
    vocab, merges = tokenizer.bpe_train(
        input_path=input_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    pr.disable() # 停止收集性能数据

    print("Profiling complete. Generating report...")

    # 将结果保存到内存流中
    s = io.StringIO()
    sortby = 'cumtime'  # 按照累计时间排序 (cumtime: 函数及其所有子函数执行的总时间)
                        # 也可以使用 'tottime' (tottime: 函数本身执行的总时间，不包括子函数)
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats() # 打印所有统计信息

    # 打印到控制台
    print(s.getvalue())

    # 也可以保存到文件，以便用 pstats 或其他工具进一步分析
    # ps.dump_stats("bpe_profiling_results.prof")
    # print("\nProfile data saved to bpe_profiling_results.prof")

    print("\nTop 10 functions by cumulative time:")
    ps.print_stats(10) # 打印前10个耗时最多的函数

"""
   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000   23.926   23.926 /Users/yfcheng/Desktop/cs336/assignment1-basics/cs336_basics/BPETokenizer.py:78(bpe_train)
        1   19.772   19.772   23.925   23.925 /Users/yfcheng/Desktop/cs336/assignment1-basics/cs336_basics/BPETokenizer.py:24(train)
 67162868    1.811    0.000    1.811    0.000 {method 'append' of 'list' objects}
 67199470    1.745    0.000    1.745    0.000 {built-in method builtins.len}
  745/744    0.262    0.000    0.467    0.001 {built-in method builtins.max}
  3656514    0.204    0.000    0.204    0.000 /Users/yfcheng/Desktop/cs336/assignment1-basics/cs336_basics/BPETokenizer.py:49(<lambda>)
  3656514    0.101    0.000    0.101    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/collections/__init__.py:601(__missing__)
       97    0.001    0.000    0.020    0.000 {method 'extend' of 'list' objects}
    27759    0.006    0.000    0.019    0.000 /Users/yfcheng/Desktop/cs336/assignment1-basics/cs336_basics/BPETokenizer.py:34(<genexpr>)
    27758    0.011    0.000    0.011    0.000 /Users/yfcheng/Desktop/cs336/assignment1-basics/cs336_basics/BPETokenizer.py:34(<listcomp>)
        1    0.000    0.000    0.006    0.006 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/regex.py:331(findall)
        1    0.005    0.005    0.005    0.005 {method 'findall' of '_regex.Pattern' objects}
      743    0.003    0.000    0.003    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/collections/__init__.py:587(__init__)
    27758    0.001    0.000    0.001    0.000 {method 'encode' of 'str' objects}
        2    0.000    0.000    0.001    0.001 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/regex.py:449(_compile)
      5/2    0.000    0.000    0.001    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:417(_parse_pattern)
     13/7    0.000    0.000    0.001    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:427(parse_sequence)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/regex.py:314(split)
      743    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/collections/__init__.py:660(update)
        3    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:803(parse_paren)
      143    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/enum.py:1525(__and__)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1116(parse_flags_subpattern)
      449    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/enum.py:1507(_get_value)
      2/1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2039(optimise)
      151    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/enum.py:688(__call__)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1102(parse_subpattern)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1417(parse_set)
       10    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1187(parse_escape)
        1    0.000    0.000    0.000    0.000 {method 'read' of '_io.TextIOWrapper' objects}
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:335(_compile_firstset)
       28    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2484(__init__)
     13/7    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3406(optimise)
       19    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:384(make_case_flags)
      9/6    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3418(pack_characters)
        1    0.000    0.000    0.000    0.000 {method 'split' of '_regex.Pattern' objects}
     47/3    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1897(compile)
      151    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/enum.py:1095(__new__)
      2/1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2133(_flatten_branches)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1483(parse_set_imp_union)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:345(_check_firstset)
      2/1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2089(pack_characters)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1374(parse_property)
      2/1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2090(<listcomp>)
      2/1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2113(_compile)
        7    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1506(parse_set_member)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/enum.py:1547(__invert__)
        1    0.000    0.000    0.000    0.000 <frozen codecs>:319(decode)
        1    0.000    0.000    0.000    0.000 {built-in method _codecs.utf_8_decode}
       18    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3500(_flush_characters)
        5    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3485(_compile)
        7    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1547(parse_set_item)
      821    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/enum.py:1377(_missing_)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/assignment1-basics/cs336_basics/BPETokenizer.py:75(<dictcomp>)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2313(_reduce_to_set)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:4316(_get_required_string)
        1    0.000    0.000    0.000    0.000 {built-in method io.open}
       13    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1850(with_flags)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1912(get_required_string)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1637(lookup_property)
      2/1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2419(max_width)
       28    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/enum.py:1367(_iter_member_by_def_)
     12/7    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2420(<genexpr>)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2146(_split_common_prefix)
      5/4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3809(optimise)
      4/3    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3829(_compile)
        2    0.000    0.000    0.000    0.000 {built-in method builtins.sorted}
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2974(pack_characters)
        5    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3589(max_width)
       52    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:4147(match)
        8    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:511(apply_quantifier)
        8    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2876(_compile)
        5    0.000    0.000    0.000    0.000 {built-in method builtins.sum}
       11    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2367(_flush_set_members)
    12/11    0.000    0.000    0.000    0.000 {method 'join' of 'str' objects}
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2106(get_firstset)
       90    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:4004(get)
        3    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:376(_flatten_code)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1626(standardise_name)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:946(parse_lookaround)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:394(make_character)
        5    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3469(get_firstset)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/assignment1-basics/cs336_basics/BPETokenizer.py:87(<genexpr>)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:410(make_property)
       15    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3590(<genexpr>)
       28    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/enum.py:1357(_iter_member_by_value_)
        2    0.000    0.000    0.000    0.000 {built-in method regex._regex.compile}
     13/7    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3402(fix_groups)
       24    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1840(make_sequence)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3894(__init__)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/regex.py:377(escape)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/locale.py:679(getpreferredencoding)
        6    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/enum.py:1514(__or__)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2969(optimise)
       87    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1847(__init__)
      2/1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2035(fix_groups)
        8    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2847(optimise)
       27    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3395(__init__)
        8    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2869(get_firstset)
        6    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/locale.py:612(setlocale)
    26/16    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1903(__hash__)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2513(_compile)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3622(rebuild)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1605(numeric_to_rational)
       10    0.000    0.000    0.000    0.000 {method 'add' of 'set' objects}
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:4215(__init__)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1395(parse_property_name)
       16    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2838(__init__)
        6    0.000    0.000    0.000    0.000 {built-in method _locale.setlocale}
        5    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3654(_handle_case_folding)
       56    0.000    0.000    0.000    0.000 {method 'get' of 'dict' objects}
       52    0.000    0.000    0.000    0.000 {method 'startswith' of 'str' objects}
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1085(parse_flags)
    26/16    0.000    0.000    0.000    0.000 {built-in method builtins.hash}
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2957(__init__)
       28    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/enum.py:117(_iter_bits_lsb)
        2    0.000    0.000    0.000    0.000 {built-in method builtins.all}
        2    0.000    0.000    0.000    0.000 {built-in method builtins.min}
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/regex.py:471(complain_unused_args)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3608(__init__)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2996(_compile)
       14    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3236(_compile)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2965(fix_groups)
        6    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2164(<genexpr>)
        1    0.000    0.000    0.000    0.000 {method '__exit__' of '_io._IOBase' objects}
       13    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:508(<listcomp>)
        8    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2852(pack_characters)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3684(max_width)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2507(get_firstset)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3213(__init__)
        8    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:559(parse_quantifier)
       12    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2158(<genexpr>)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3922(_compile)
        3    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:4186(expect)
        8    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2907(max_width)
        1    0.000    0.000    0.000    0.000 <frozen codecs>:309(__init__)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3058(optimise)
       26    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/enum.py:1374(<lambda>)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2031(__init__)
        8    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2844(fix_groups)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3230(get_firstset)
        2    0.000    0.000    0.000    0.000 <frozen importlib._bootstrap>:1207(_handle_fromlist)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:4070(get_while)
       29    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2504(optimise)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:4234(open_group)
       30    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1870(fix_groups)
       28    0.000    0.000    0.000    0.000 {built-in method builtins.chr}
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3992(__init__)
       27    0.000    0.000    0.000    0.000 {built-in method builtins.ord}
       17    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3227(optimise)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3087(_compile)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3035(__del__)
        5    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1906(__eq__)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/enum.py:1453(<listcomp>)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3626(get_firstset)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3049(__init__)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2074(_add_precheck)
       11    0.000    0.000    0.000    0.000 {method 'isspace' of 'str' objects}
        8    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1631(<genexpr>)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2540(max_width)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1069(parse_flag_set)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3200(__init__)
        9    0.000    0.000    0.000    0.000 {method 'pop' of 'list' objects}
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:4260(close_group)
       10    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1876(pack_characters)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:4277(_check_group_features)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3032(get_required_string)
        3    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3936(max_width)
        8    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1900(is_empty)
        2    0.000    0.000    0.000    0.000 {built-in method _locale.getencoding}
        9    0.000    0.000    0.000    0.000 {method 'items' of 'dict' objects}
        4    0.000    0.000    0.000    0.000 {method 'split' of 'str' objects}
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:4190(at_end)
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3203(_compile)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2993(has_simple_start)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3055(fix_groups)
        4    0.000    0.000    0.000    0.000 {method 'upper' of 'str' objects}
        4    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}
        4    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3709(__del__)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        2    0.000    0.000    0.000    0.000 {built-in method __new__ of type object at 0x100dd95f8}
        2    0.000    0.000    0.000    0.000 {method 'setdefault' of 'dict' objects}
        3    0.000    0.000    0.000    0.000 {method 'discard' of 'set' objects}
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/regex.py:476(<setcomp>)
        1    0.000    0.000    0.000    0.000 {method 'clear' of 'set' objects}
        1    0.000    0.000    0.000    0.000 <frozen codecs>:260(__init__)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3065(pack_characters)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1885(can_be_affix)
        2    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/regex.py:642(<genexpr>)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:2902(__eq__)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3939(get_required_string)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:1894(has_simple_start)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3111(max_width)
        1    0.000    0.000    0.000    0.000 /Users/yfcheng/Desktop/cs336/.conda/lib/python3.11/site-packages/regex/_regex_core.py:3919(has_simple_start)
"""

