import torch
import torch.nn.functional as F


class OnlineEWC:
    """Online diagonal empirical Fisher; constant memory across task boundaries."""

    def __init__(self, strength=100.0, decay=1.0, scope="all"):
        if strength < 0 or not 0 <= decay <= 1 or scope not in {"all", "quantum"}:
            raise ValueError("Invalid EWC strength, decay, or scope")
        self.strength, self.decay, self.scope = strength, decay, scope
        self.fisher, self.anchor = {}, {}

    def parameters(self, model):
        return {n: p for n, p in model.named_parameters() if p.requires_grad and (self.scope == "all" or n == "alignment.angles")}

    def penalty(self, model):
        params = self.parameters(model)
        zero = next(model.parameters()).new_zeros(())
        return self.strength * .5 * sum(
            (self.fisher[n].to(p) * (p - self.anchor[n].to(p)).square()).sum()
            for n, p in params.items() if n in self.fisher
        ) + zero

    def consolidate(self, model, loader, device, ignore_index=255, max_images=16, pixels_per_image=4, seed=42):
        if self.strength == 0:
            return
        if max_images < 1 or pixels_per_image < 1:
            raise ValueError("Fisher sampling budgets must be positive")
        params = self.parameters(model)
        if not params:
            raise ValueError("EWC scope selected no trainable parameters")
        fisher = {n: torch.zeros_like(p, device="cpu") for n, p in params.items()}
        rng = torch.Generator().manual_seed(seed)
        prior_mode = model.training
        model.eval()
        count, seen = 0, 0
        try:
            for batch in loader:
                for image, mask in zip(batch["image"], batch["mask"]):
                    if seen >= max_images:
                        break
                    seen += 1
                    locations = (mask.flatten() != ignore_index).nonzero().flatten()
                    if not len(locations):
                        continue
                    chosen = locations[torch.randperm(len(locations), generator=rng)[:pixels_per_image]]
                    logits = model(image[None].to(device))[0].flatten(1)
                    targets = mask.flatten().to(device)
                    for i, pixel in enumerate(chosen.tolist()):
                        nll = F.cross_entropy(logits[:, pixel][None], targets[pixel][None])
                        grads = torch.autograd.grad(nll, tuple(params.values()), retain_graph=i + 1 < len(chosen), allow_unused=True)
                        for (name, _), grad in zip(params.items(), grads):
                            if grad is not None:
                                fisher[name] += grad.detach().cpu().square()
                        count += 1
                if seen >= max_images:
                    break
            if count == 0:
                raise ValueError("No valid labeled pixels available for Fisher estimation")
            self.fisher = {n: f / count + self.decay * self.fisher.get(n, 0) for n, f in fisher.items()}
            if any(not torch.isfinite(f).all() for f in self.fisher.values()):
                raise FloatingPointError("Non-finite Fisher statistics")
            self.anchor = {n: p.detach().cpu().clone() for n, p in params.items()}
        finally:
            model.train(prior_mode)

    def state_dict(self):
        return {"strength": self.strength, "decay": self.decay, "scope": self.scope, "fisher": self.fisher, "anchor": self.anchor}

    def load_state_dict(self, state):
        self.strength, self.decay, self.scope = state["strength"], state["decay"], state["scope"]
        self.fisher, self.anchor = state["fisher"], state["anchor"]
