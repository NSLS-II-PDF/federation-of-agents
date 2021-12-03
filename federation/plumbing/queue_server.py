from queue import Queue
from event_model import RunRouter
from bluesky_adaptive.utils import extract_event_page
from bluesky_adaptive.recommendations import NoRecommendation


def index_reccomender_factory(
    adaptive_object,
    sample_index_key,
    sample_data_key,
    *,
    queue=None,
    cache_callback=None,
):
    if queue is None:
        queue = Queue()

    if cache_callback is None:
        prelim_callbacks = ()
    else:
        prelim_callbacks = [
            cache_callback,
        ]

    def callback(name, doc):
        """Assumes the start doc gives you the sample location,
        and the event_page gives quality info. The current index is updated at the start
        But the Agent quality matrix is only updated at tell."""
        # TODO: Validate the assumptions on formats
        # TODO: Update queue signatures from .put to ...?
        print(f"callback received {name}")

        if name == "start":
            current_index = doc[sample_index_key]
            adaptive_object.tell(x=current_index)

        elif name == "event_page":
            data = extract_event_page(
                [
                    sample_data_key,
                ],
                payload=doc["data"],
            )
            adaptive_object.tell(y=data)

            try:
                next_point = adaptive_object.ask(1)
            except NoRecommendation:
                queue.put(None)
            else:
                queue.put({sample_index_key: next_point})
        else:
            print(f"Document {name} is not handled")

    rr = RunRouter([lambda name, doc: ([prelim_callbacks, callback], [])])
    return rr, queue
