"""Module for handling edge detection and summarizing"""


class AllEdgesHandler(CalculationHandler):
    """Handler for all_edges. Uses numpy.load"""
    _db_field_path = 'all_edges_filename'
    _name = 'all_edges'
    
    def load_data(self):
        filename = self.get_path
        try:
            data = np.load(filename)
        except IOError:
            raise IOError("no all_edges found at %s" % filename)
        return data

class EdgeSummaryHandler(CalculationHandler):
    _db_field_path = 'edge_summary_filename'
    _name = 'edge_summary'

