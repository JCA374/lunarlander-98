# Episode 03: Inside the pilot, and why we test so much

Hello. You asked for a film that explains the bake off, and a podcast to go with it. The film is on the website now. This episode is the version for your ears. It explains what the pilot actually is, why the winner won, and why there are so many tests around every number I give you.

Let's start with the pilot. Fifty times a second it gets sixteen numbers. Eight come straight from the game. Where it is, how fast it moves, how it is tilted, and whether its legs touch the ground. The other eight are sensors. Where the boulder is. A point to aim for beside it. And a wind estimate. The pilot cannot see the wind. It can only feel a sideways push. But its own engines push it sideways too. So the sensor subtracts the push the pilot knows it just made. What is left is the wind.

You also asked to see the wind. So every replay on the site now draws it. A blue arrow at the top is the true wind, the one the pilot never sees. An orange arrow below it is the pilot's own guess. Faint dust streaks drift across the sky with the gusts. In a test flight of the winner, the two arrows moved together, with a correlation of zero point nine four. When you watch the lander lean into a gust, you can now see why.

The pilot itself is a neural network. Sixteen numbers go in, four come out, one per action. Do nothing, left engine, main engine, right engine. It fires the highest one. In between are two layers of a hundred and twenty eight units. Each unit adds up its inputs, each multiplied by a weight, and squashes the sum to between minus one and one. That is all. There are about nineteen thousand weights. Training means finding good values for them.

Three ways of finding those weights met in the bake off. The winner was reinforcement learning from scratch. During training, the network does not just pick an action. It rolls weighted dice. A second network, the critic, guesses how well the flight will end from each moment. After a batch of flights, each action that turned out better than the critic expected becomes a little more likely. Each action that turned out worse becomes a little less likely. And every step of every flight teaches it something.

The copy of the hand written pilot was too sure of itself. It almost never tried anything new, so there was nothing to learn from. The evolution strategy judged each version on thirty two windy flights. That is too few to separate skill from luck.

Now the tests. I want to explain why they exist, because each one guards against a mistake that really happened.

First, the task is frozen. A test checks the fingerprint of the code that defines the world. If one character changes, the test fails. Without it, we could reach ninety eight percent on an easier world and never notice.

Second, the same seed must give the same flight. In the film you can watch two separate simulators fly the same seed side by side. They match to the last decimal. In an earlier project, the wind took its starting point from a hidden random generator. Then one seed did not mean one flight, and no comparison was fair.

Third, no peeking. A test rebuilds every input the pilot got, using only what it observed and its own actions. Then it flies the pilot with no access to the simulator at all. The flight is identical. So the pilot is not reading the hidden wind.

Fourth, separate seeds. Training, checks, choosing champions, and the final exam each have their own range of seed numbers. The exam's range has never been touched. A test makes sure the ranges never overlap.

And fifth, the winner's curse. Near the end of the bake off, the winner's best quick check said ninety seven percent. Its average check said about ninety four. On five hundred fresh flights it scored ninety one point four. Pick the best of many noisy numbers, and you pick the luckiest. That is why the goal needs a thousand fresh flights, twice.

Where are we now? The main run continues the winner. So far its checks sit around ninety two and a half percent. Its failures are mostly boulder touches, especially when the boulder floats off to the side. Next episode: when something moves.
