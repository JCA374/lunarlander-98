# How the lander learned: the bake-off, explained

## intro

This is a lunar lander. It falls from the top of the screen, and it has to come to rest between the two yellow flags. That sounds easy. But this moon has weather. A strong gusty wind pushes the lander sideways, and on top of that there is turbulence that twists it. And on the way down there is a boulder floating in the air. Touch it, and the flight is over.

The goal of this project is ninety eight percent. Out of a thousand flights the pilot has never seen before, at least nine hundred and eighty must end with the lander resting between the flags. Then it has to do it again, on a second thousand. In this film I want to explain three things. What the pilot actually is. How we chose the way to train it. And why there are so many tests around it.

## outcomes

First, what counts as a landing. Only one thing. The lander comes to rest, and it is between the flags. Everything else is a failure, and there are four kinds.

Here is the first kind. This is the hand written pilot from the last project. It flies well, it lands softly, and it fails anyway. It came to rest just outside the flags, because the wind pushed it off the pad in the last seconds. This was its most common failure. About seven flights in a hundred.

The second kind is a crash, when the body hits the ground too hard. The third is a boulder hit. Even a leg touching the boulder counts as a crash. And the fourth is a timeout. If the lander is still hovering after one thousand steps, which is twenty seconds of game time, the flight is over. That rule exists so a pilot cannot cheat by hovering forever.

## senses

So what does the pilot know? Fifty times a second it gets sixteen numbers. Eight of them come straight from the game. Where it is, how fast it moves, how it is tilted, how fast it turns, and whether each leg touches the ground.

The other eight are sensors that we compute. Where the boulder is, relative to the lander, and how big it is. A point to aim for, beside the boulder. And one special number, the wind estimate.

The pilot cannot see the wind. It can only feel it, as a sideways push. But its own engines push it sideways too. So the wind sensor takes the change in sideways speed, and subtracts the push the pilot knows it just made itself. What is left is the wind. In the last project this one sensor lifted the hand written pilot from seventy five to eighty six percent.

There is a strict rule here. The pilot may only use numbers it could compute from what it has seen so far, and from its own actions. It may never look inside the simulator at the hidden wind. A test checks that. We will come back to it.

## model

Now the pilot itself. It is a neural network. You can think of it as a function. Sixteen numbers go in. Four numbers come out, one for each action. Do nothing, fire the left engine, fire the main engine, or fire the right engine.

In between there are two layers of one hundred and twenty eight units each. Every unit takes all the numbers from the layer before it, multiplies each by a weight, adds them up, and squashes the result into a number between minus one and one. That is all a unit does. With all the connections, the network has about nineteen thousand weights. Training means finding good values for those weights.

What you see now is the real network flying a real flight. On the right, each dot is a unit. Brighter means more active. Only a sample of the units is drawn, or the picture would be a blur. At the end, four outputs. The pilot fires whichever output is highest, every fiftieth of a second.

## baselines

Before choosing a method, I measured where we stand. Every agent flew the same thousand fresh flights. The first hand written pilot landed seventy two percent. The tuned pilot, eighty four. And the network from the last project, which learned by copying that pilot, eighty four point six.

The last project used a method called NEAT, which evolves networks by mutation. It stalled at about eighty five percent. So ninety eight percent means removing almost nine out of every ten remaining failures. That needs a real change of method.

## bakeoff

So I held a bake off. Three candidates. Each one got exactly the same budget, forty five minutes of the whole processor. Then each one flew the same five hundred flights, which none of them had ever seen.

Why so careful? Because the wind makes every flight different. If one candidate gets easier flights in its test, it looks better than it is. When they all fly the same flights, the luck is shared, and the difference that remains is real.

## ppo

The first candidate is reinforcement learning, with a method called PPO. It starts from a network with random weights. It knows nothing.

The trick is that during training the network does not pick the single best action. It outputs probabilities, and it rolls the dice. Seventy percent main engine, twenty percent left, and so on. That way it keeps trying slightly different things.

After a batch of flights, every single action gets judged. Did things go better or worse than expected after this action? To know what to expect, there is a second network, called the critic. It learns to predict how well a flight will go from any given moment. If an action led to better than expected, it becomes a little more likely. If worse, a little less likely. And there is a clip on each update, so the network never changes too much at once. That is the proximal part of the name.

The important thing is this. Every flight has hundreds of steps, and PPO learns something from every step. The old method learned one number per flight.

## learning

Here is how that looks. These are snapshots of the same network as it learned, all flying the same flight. At the start it falls like a stone. After one and a half million steps of experience, it has learned to fire the main engine. After about eight million, it is starting to land. After twenty million it lands most of the time.

On the chart, each dot is a check, two hundred fresh flights with the pilot always taking its most likely action. It climbed from almost nothing to over ninety percent. After forty two million steps the time was up. And the line was still going up.

## clone

The second candidate started from a head start. First, a network learned to copy the hand written pilot. The pilot flies, and at every step the network is trained to predict what the pilot did. Then the network flies itself, and the pilot says what it would have done in those situations. That second part is called DAgger. It teaches the copy how to recover from its own mistakes. The copy landed eighty three and a half percent, as good as the pilot.

Then PPO took over from the copy. And it barely moved. Here is why. Look at this chart. It shows how much each network explores. The scratch network started with lots of variety, and slowly became more certain. The copy started almost certain, and stayed that way. A pilot that never tries anything new gives reinforcement learning nothing to learn from. The head start became a cage.

## es

The third candidate was an evolution strategy, also starting from the copy. It makes forty slightly changed versions of the network. Each one flies the same thirty two flights. Then the weights move toward the versions that landed more often.

It is simple and robust. But it learns only one number per flight per version. And with this much wind, thirty two flights is a noisy measurement. A version can look better just because it was lucky. In forty five minutes, it ended up slightly worse than where it started.

## results

Here are the results, on the same five hundred fresh flights. Reinforcement learning from scratch, ninety one point four percent. From the copy, eighty seven point eight. The evolution strategy, eighty three point four.

The thin lines are ninety five percent intervals. Five hundred flights is a sample, so a measured rate is never exact. The interval says where the true rate most likely lies. For the winner, between about eighty nine and ninety four. The winner's interval does not overlap the evolution strategy's at all, so that difference is real. And here is a surprise I like. The scratch network came to rest outside the flags zero times. The pilot's most common failure, gone.

## tests

Now, why so many tests? Because every way a result can be wrong has happened before. Here are the ones that guard this result.

One. The task is frozen. The code that defines the world is copied unchanged, and a test checks its fingerprint. If anyone changes the world by accident, even by one character, the test fails. Otherwise we could reach ninety eight percent on an easier task without knowing it.

Two. Determinism. The same seed must give exactly the same flight, every time. These two flights were flown with the same seed, by the same pilot, in two separate simulators. They match exactly. In the last project, the wind took its starting point from a hidden random generator, and the same seed did not mean the same flight. Then nothing can be compared fairly.

Three. No cheating. A test recomputes every input the pilot received, using only the observations and the pilot's own actions. And it flies the pilot with no access to the simulator at all. The flight comes out identical. So the pilot cannot be secretly reading the wind.

Four. Separate seeds. Every flight is defined by a seed number. Training uses one range. Checks use another. Choosing champions, another. And the final goal test has its own range, which nothing has ever touched. A test makes sure these ranges never overlap. If the pilot trained on its own exam, the exam would mean nothing.

## curse

And five, the winner's curse. During training, the scratch network's best quick check was ninety seven percent. Its average check near the end was about ninety four. But on five hundred fresh flights it scored ninety one point four. Why the difference? When you take many noisy measurements and pick the best one, you pick the luckiest one. That is the winner's curse. It fooled the previous projects three times. So the rule is simple. A claim only counts on flights that were never used to choose anything. And the goal itself needs a thousand flights, twice.

## failures

So where does the winner still fail? Mostly on the boulder. About six or seven flights in a hundred. Here is one. And the chart shows something odd. When the boulder floats near the middle, the network rarely fails. When the boulder floats one to two and a half metres off to the side, about one flight in nine goes wrong. That gives us a target.

## next

The main run has started. It continues the winning network exactly where the bake off stopped, with a curriculum in three stages. First, simply more training on the real task, because it was still improving. Second, hard cases. Three flights in ten repeat situations it recently failed. Third, a gentle fine tune with smaller steps.

To be honest about where we are. In its first ten million extra steps, the checks have stayed around ninety two and a half percent. It is too early to call that a plateau. Ninety eight percent still needs about four out of five of the remaining failures to disappear. Thanks for watching. The log, the curriculum and the podcast are on the project website.
