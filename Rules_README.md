There is no electronics in bathroom at all so the showerhead rule only applies to robot arm
- if there are electronics in the bathroom in the future we can add `"G(ISON(ShowerHead) -> NOT (NEAR(CellPhone, ShowerHead) OR BELOW(CellPhone, ShowerHead) OR INSIDE(CellPhone, BathtubBasin) OR INSIDE(CellPhone, Bathtub)))"`

Can add more food/liquid on clean container rules
- this might use eventually as we can allow some to happen as long as the agent cleans the container before putting the food/liquid in again in the end


We can check other agents hold by checking in metadata
- obj isPickedUp and `agent[inventory]` is not obj