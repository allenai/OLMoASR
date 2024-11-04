import pytest
import os
from scripts.data.filtering.data_filter import DataFilter, FilterFunc, DataReader
import ray


class DataFilterTest(DataFilter):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def base_filter(self, filter_func: FilterFunc):
        ray.init()
        print("Start reading binary files")
        ds = ray.data.read_binary_files(
            paths=self.data_dir, file_extensions=["srt", "vtt"], include_paths=True
        ).map(DataReader.bytes_to_text)

        print("Finish reading binary files")
        total = ds.count()
        
        print("Start filtering")
        ds = ds.filter(filter_func)
        filtered = ds.count()
        removed = total - filtered
        
        return (removed, total)

class TestFilters:
    @staticmethod
    def base_test(data_dir, filter_func):
        data_filter = DataFilterTest(data_dir=data_dir)
        removed, total = data_filter.base_filter(filter_func)
        assert total == len(os.listdir(data_dir))
        assert removed == total
        
    def test_not_lower(self):
        data_dir = "data/tests/filter/not_lower"
        filter_func = FilterFunc.not_lower
        TestFilters.base_test(data_dir, filter_func) 
        
    def test_not_upper(self):
        data_dir = "data/tests/filter/not_upper"
        filter_func = FilterFunc.not_upper
        TestFilters.base_test(data_dir, filter_func)
    
    def test_only_mixed(self):
        data_dir = "data/tests/filter/only_mixed"
        filter_func = FilterFunc.only_mixed
        TestFilters.base_test(data_dir, filter_func)
    
    def test_no_repeat(self):
        data_dir = "data/tests/filter/no_repeat"
        filter_func = FilterFunc.no_repeat
        TestFilters.base_test(data_dir, filter_func)
    
    def test_min_comma_period(self):
        data_dir = "data/tests/filter/min_comma_period"
        filter_func = FilterFunc.min_comma_period
        TestFilters.base_test(data_dir, filter_func)
    
    def test_min_comma_period_mixed_no_repeat(self):
        data_dir = "data/tests/filter/min_comma_period_mixed_no_repeat"
        filter_func = FilterFunc.min_comma_period_mixed_no_repeat
        TestFilters.base_test(data_dir, filter_func)
    

    