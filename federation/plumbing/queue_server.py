from bluesky_adaptive.utils import extract_event_page
from bluesky_adaptive.recommendations import NoRecommendation


def index_reccomender_factory(
    *,
    adaptive_object,
    sample_index_key,
    sample_data_key,
    queue_server,
    # TODO: Add more sensible defaults for these args.
    mv_kwargs=None,
    count_args=(),
    count_kwargs=None,
):
    if mv_kwargs is None:
        mv_kwargs = {}
    if count_kwargs is None:
        count_kwargs = {}

    def callback(name, doc):
        """Assumes the start doc gives you the sample location,
        and the event_page gives quality info. The current index is updated at the start
        But the Agent quality matrix is only updated at tell."""
        # TODO: Validate the assumptions on formats
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
                queue_server.queue_item_add(None)
            else:
                response = queue_server.queue_item_add(
                    item_name="mv",
                    item_args=[next_point],
                    item_kwargs=mv_kwargs,
                    item_type="plan",
                )
                if response.json()["success"] is False:
                    raise RuntimeError("Queue Server failed to add item for mv plan")
                response = queue_server.queue_item_add(
                    item_name="count",
                    item_args=[*count_args],
                    item_kwargs=count_kwargs,
                    item_type="plan",
                )
                if response.json()["success"] is False:
                    raise RuntimeError("Queue Server failed to add item for count plan")
        else:
            print(f"Document {name} is not handled")

    return callback
