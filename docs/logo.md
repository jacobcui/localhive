# Logo

<img src="logo.svg" alt="localhive logo" width="96">

## The idea

A **hexagonal memory budget, filled with the models that fit it.**

The outer hexagon is your machine. The seven golden cells are the models
`list-models` found that fit inside it. Hexagons are the densest way to pack
things into a fixed space, which is exactly what the tool does with your RAM —
so the shape is the product, not decoration.

It also happens to read as a hive, which is where the name comes from.

## Files

| File | Use |
| --- | --- |
| [logo.svg](logo.svg) | Primary mark. Light backgrounds. |
| [logo-dark.svg](logo-dark.svg) | Light body, for dark backgrounds. |
| [logo-mono.svg](logo-mono.svg) | One colour. Uses `currentColor`; cells are knocked out with a mask, so it works on any background and in print. |
| [logo-lockup.svg](logo-lockup.svg) | Mark + wordmark + tagline, horizontal. |
| [favicon.svg](favicon.svg) | Tighter cut with no outer padding, fatter cells so the lattice survives 16 px. |
| [social-preview.png](social-preview.png) | 1280x640 card for link unfurls. GitHub only lets you set this in the web UI — see below. |

All are hand-written SVG — no embedded bitmaps, no fonts required for the mark
itself. Each is under 2.5 KB.

## Geometry

The mark is exact, not eyeballed. For a pointy-top honeycomb:

* a 7-cell flower (1 centre + 6 ring) at radius `r`
* spans `sqrt(3) * 1.5 * r` horizontally
* the outer hexagon's apothem is `sqrt(3)/2 * R`

So the largest flower that fits inside the outer hexagon without crowding the
border is `r <= R/3`. The mark uses `r = 0.333 * R * 0.86`, leaving deliberate
breathing room.

Cells are inset to 90% of their radius; the dark body showing through the gaps
draws the lattice, rather than a stroke. That keeps the silhouette solid at
small sizes instead of turning into outline mush.

## Colours

| Role | Hex |
| --- | --- |
| Honey gradient (start) | `#fcd34d` |
| Honey gradient (mid) | `#f59e0b` |
| Honey gradient (end) | `#ea580c` |
| Body (light mode) | `#111827` |
| Body (dark mode) | `#e5e7eb` |
| Tagline grey | `#6b7280` |

## Usage

```html
<img src="docs/logo.svg" alt="localhive" width="112" height="112">
```

Minimum size: **16 px**. Below that the lattice closes up — use
[logo-mono.svg](logo-mono.svg) or the plain hexagon instead.

One-colour contexts:

```html
<span style="color:#111827"><img src="docs/logo-mono.svg" width="64"></span>
```

Please don't recolour the mark, stretch it, or add effects. Scaling
proportionally is always fine.

## Social preview

GitHub has no API for the repository social preview image, so it must be set
by hand once:

1. Go to **Settings → General → Social preview**
2. Upload [social-preview.png](social-preview.png)

That image is what shows when the repo is shared on Slack, Discord, X or
LinkedIn. Without it, those links unfurl as a plain grey box.
