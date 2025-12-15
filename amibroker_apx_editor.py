import xml.etree.ElementTree as ET
from datetime import datetime
import os

class AmiBrokerAnalysis:
    def __init__(self, apx_path):
        self.apx_path = apx_path
        self.tree = ET.parse(apx_path)
        self.root = self.tree.getroot()

    def set_date_range(self, from_date, to_date):
        """Set analysis date range
        Args:
            from_date (datetime): Start date
            to_date (datetime): End date
        """
        # Format dates as required by AmiBroker
        from_str = from_date.strftime("%Y-%m-%d %H:%M:%S") if from_date else None
        to_str = to_date.strftime("%Y-%m-%d %H:%M:%S") if to_date else None

        # Update date ranges in both General and BacktestSettings
        for section in ['General', 'BacktestSettings']:
            elem = self.root.find(section)
            if elem is not None:
                from_elem = elem.find('FromDate')
                to_elem = elem.find('ToDate')
                if from_elem is not None and from_str is not None:
                    from_elem.text = from_str
                if to_elem is not None and to_str is not None:
                    to_elem.text = to_str

                # Also update backtest range dates if present
                if section == 'BacktestSettings':
                    from_elem = elem.find('BacktestRangeFromDate')
                    to_elem = elem.find('BacktestRangeToDate')
                    if from_elem is not None and from_str is not None:
                        from_elem.text = from_str
                    if to_elem is not None and to_str is not None:
                        to_elem.text = to_str

    def set_filter_categories(self, include_categories=None, exclude_categories=None):
        """Set include/exclude filter categories
        Args:
            include_categories (list): List of category IDs to include
            exclude_categories (list): List of category IDs to exclude
        """
        if include_categories:
            for filter_elem in self.root.findall('.//IncludeFilter'):
                for i, cat_id in enumerate(include_categories):
                    cat_elem = filter_elem.find(f'Category{i}')
                    if cat_elem is not None:
                        cat_elem.text = str(cat_id)

        if exclude_categories:
            for filter_elem in self.root.findall('.//ExcludeFilter'):
                for i, cat_id in enumerate(exclude_categories):
                    cat_elem = filter_elem.find(f'Category{i}')
                    if cat_elem is not None:
                        cat_elem.text = str(cat_id)

    def set_backtest_settings(self, **kwargs):
        """Update backtest settings
        Args:
            initial_equity (float): Initial equity amount
            commission (float): Commission value
            max_positions (int): Maximum number of positions
            etc.
        """
        backtest = self.root.find('BacktestSettings')
        if backtest is not None:
            for key, value in kwargs.items():
                # Convert camelCase to element names
                elem_name = ''.join(x.capitalize() for x in key.split('_'))
                elem = backtest.find(elem_name)
                if elem is not None:
                    elem.text = str(value)

    def save(self, output_path=None):
        """Save changes to file"""
        if output_path is None:
            output_path = self.apx_path
        self.tree.write(output_path, encoding='ISO-8859-1', xml_declaration=True)

def modify_analysis(apx_path, **kwargs):
    """Helper function to modify analysis file
    
    Example usage:
        modify_analysis('analysis.apx',
            date_range=(start_date, end_date),
            include_categories=[1,2,3],
            exclude_categories=[4,5],
            backtest_settings={'initial_equity': 1000000}
        )
    """
    analysis = AmiBrokerAnalysis(apx_path)
    
    if 'date_range' in kwargs:
        from_date, to_date = kwargs['date_range']
        analysis.set_date_range(from_date, to_date)
        
    if 'include_categories' in kwargs:
        analysis.set_filter_categories(include_categories=kwargs['include_categories'])
        
    if 'exclude_categories' in kwargs:
        analysis.set_filter_categories(exclude_categories=kwargs['exclude_categories'])
        
    if 'backtest_settings' in kwargs:
        analysis.set_backtest_settings(**kwargs['backtest_settings'])
        
    analysis.save()
