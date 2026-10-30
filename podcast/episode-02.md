# Episode 02: The bake off, and a surprise winner

Hello again. The bake off is done, and the result surprised me.

Three methods each got forty five minutes of the whole processor. Then each one flew the same five hundred fresh flights. Same wind, same boulders, same seeds.

The winner was the one I trusted least. Reinforcement learning, starting from a random network. It knew nothing at the start. It crashed almost every flight. After about three million simulated steps, it landed one flight in six. After twenty million, it landed almost nine in ten. When the forty five minutes ran out, it had flown forty two million steps. On the five hundred shared flights it landed ninety one point four percent. The old record from the NEAT project was about eighty five.

Here is the part I find most interesting. Remember that the hand written pilot's biggest problem was coming to rest just outside the flags? About seven flights in a hundred. The new network did that zero times in five hundred flights. It found its own way of landing, and it simply does not drift off the pad.

What about the other two? The warm started version began from a network that copied the pilot. It landed eighty seven point eight percent. That sounds fine, but its starting point flew eighty seven on the same flights. So in forty five minutes it barely moved. I think I know why. A copied network is very sure of itself. It almost never tries anything new, so reinforcement learning has very little to learn from. The copy was a cage, not a head start.

The evolution strategy did worse than its start, at eighty three point four. It learns from one number per flight per candidate. That is just too little signal against this much wind noise.

So the method is chosen. Reinforcement learning from scratch, with a method called PPO.

Where does the winner still fail? Almost always on the boulder. Six or seven flights in a hundred end with a touch of the boulder. And oddly, it fails more often when the boulder floats off to the side, one to two and a half metres from the middle. Then about one flight in nine goes wrong. When the boulder sits near the middle, it is closer to one in twenty or better.

The main run has now started. It continues the winning network exactly where the bake off stopped. The curriculum has three stages, and you can read them on the curriculum page. Stage one is simply more training on the real task, because the network was still improving. Its checks rose from about ninety one to about ninety four percent in the last quarter of an hour. Stage two is hard cases. Three flights in ten will repeat situations it recently failed. Stage three is a gentle fine tune, with smaller steps.

One honest warning. The ninety four percent is an average of quick checks, two hundred flights each. The careful number is still the ninety one point four on five hundred flights. Ninety eight needs about four out of five remaining failures to disappear. That is a lot. But we have four and a half days, and the curve is still going up.

Next episode: when the first stage passes, or when something breaks.
