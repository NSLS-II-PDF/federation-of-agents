import argparse
from databroker._drivers.msgpack import BlueskyMsgpackCatalog
from event_model import RunRouter
from suitcase.msgpack import Serializer
from pathlib import Path
import pprint
from federation.quality.base import SequentialAgent, MarkovAgent
from federation.plumbing.queue_server import index_reccomender_factory
from bluesky.callbacks.zmq import RemoteDispatcher as ZmqRemoteDispatcher
from exp_queueclient import BlueskyHttpserverSession
import logging


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--document-cache", type=Path, default=None)
    arg_parser.add_argument("--agent", type=str, default="sequential")
    arg_parser.add_argument("-n", "--n-samples", type=int, default=30)
    arg_parser.add_argument("--seed", type=int, default=1234)

    # TODO: Update default server arguments
    arg_parser.add_argument("--zmq-host", type=str, default="xf28id1-ca1")
    arg_parser.add_argument("--zmq-subscribe-port", type=int, default=5578)
    arg_parser.add_argument("--zmq-subscribe-prefix", type=str, default="rr")

    arg_parser.add_argument(
        "-u", "--bluesky-httpserver-url", type=str, default="http://localhost:60610"
    )

    args = arg_parser.parse_args()
    pprint.pprint(vars(args))

    zmq_dispatcher = ZmqRemoteDispatcher(
        address=(args.zmq_host, args.zmq_subscribe_port),
        prefix=args.zmq_subscribe_prefix.encode(),
    )

    ####################################################################
    # CHOOSE YOUR FIGHTER
    agent = {
        "sequntial": SequentialAgent(args.n_samples),
        "markov": MarkovAgent(args.n_samples, max_quality=3, seed=1234),
    }[args.agent]
    ####################################################################

    if args.document_cache is not None:
        cat = BlueskyMsgpackCatalog(str(args.document_cache / "*.msgpack"))
        xs = []
        ys = []
        for uid in cat:
            h = cat[uid]
            xs.append(h.metadata["start"]["sample_number"])
            ys.append(h.primary.read())
        agent.tell_many(xs, ys)
        cache_callback = Serializer(args.document_cache, flush=True)
    else:
        cache_callback = None

    with BlueskyHttpserverSession(
        bluesky_httpserver_url=args.bluesky_httpserver_url
    ) as session:
        ####################################################################
        # ENSURE THESE KEYS AND QUEUE ARE APPROPRIATE
        index_callback, _ = index_reccomender_factory(
            adaptive_object=agent,
            sample_index_key="sample number",
            sample_data_key="data",
            queue_server=session,
        )
        ####################################################################

        rr = RunRouter([lambda name, doc: ([cache_callback, index_callback], [])])
        zmq_dispatcher.subscribe(rr)

        logging.debug(
            f"ADAPTIVE CONSUMER LISTENING ON {args.zmq_subscribe_prefix.encode()}"
        )
        zmq_dispatcher.start()
