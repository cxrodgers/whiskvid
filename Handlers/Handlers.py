import os
import pandas
from base import *
from WhiskersTableHandler import WhiskersTableHandler
from EdgesHandlers import AllEdgesHandler, EdgeSummaryHandler
from ColorizedWhiskerEnds import ColorizedWhiskerEndsHandler
from TacHandler import TacHandler, ClusteredTacHandler

class MonitorVideoHandler(CalculationHandler):
    """Handler for a monitor video generated by WhiskiWrap trace.
    
    Unlike other handlers, this data is a side effect of trace, so
    this would be set by another calculation.
    
    load_data is not implemented
    """
    _db_field_path = 'monitor_video'
    _name = 'monitor_video'
    
    # Override because it's not just _name
    @property
    def new_path(self):
        return self.video_session.name + '.mkv'

## Handlers for data that haven't been incorporated into the database yet
# Need to do this, so that we can easily check which have been generated
# For now we hard-code the path
class CalculationHandlerWithoutDb(CalculationHandler):
    """Overload for data not in db
    
    """
    @property
    def get_path(self):
        """Assumes self.new_path is the name, checks if it exists, returns

        Raises IOError if the file doesn't exist
        """
        full_filename = os.path.join(self.video_session.session_path,
            self.new_path)
        if not os.path.exists(full_filename):
            raise IOError("file does not exist: %s" % full_filename)
        return full_filename
    
    def set_path(self):
        raise NotImplementedError(
            "set_path not available because this calculation is not in the db")

class MaskedWhiskerEndsHandler(CalculationHandlerWithoutDb):
    """Whisker ends after follicle mask"""
    _name = 'masked_whisker_ends'
    
    # Override because it's not just _name
    @property
    def new_path(self):
        return 'mwe'

class VideoTrackedWhiskersHandler(CalculationHandlerWithoutDb):
    """Tracked whiskers video. Side effect of trace"""
    _name = 'video_tracked_whiskers'

    # Override because it's not just _name
    @property
    def new_path(self):
        return self.video_session.name + '.tracked_whiskers.mkv'
    
class VideoColorizedWhiskersHandler(CalculationHandlerWithoutDb):
    """Tracked whiskers video. Side effect of colorizing"""
    _name = 'video_colorized_whiskers'
    
    # Override because it's not just _name
    @property
    def new_path(self):
        return self.video_session.name + '_colorized_video.mp4'
