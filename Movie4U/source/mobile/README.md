# Movie4U mobile demo

Expo / React Native / TypeScript app for comparing KG-only and KG + GraphSAGE recommendations. Uses existing anonymized MovieLens profiles; it does not implement account creation or live retraining.

## Run

1. Run the Python pipeline and API (see the project README).
2. `npm install`
3. Copy `.env.example` to `.env` and set `EXPO_PUBLIC_API_URL` to the backend URL.
4. `npm start` and open with Expo Go, or `npm run android` / `npm run ios`.
5. `npm run web` provides an optional browser preview.

Use your computer's LAN IP on a physical phone, `http://10.0.2.2:8000` on an Android emulator, or `http://localhost:8000` for web/iOS simulator. The phone and computer must be able to reach each other. Bind the API to `0.0.0.0` for LAN access; this is a development server, not a production deployment.

`npm run typecheck` checks TypeScript. `npx expo export --platform web` checks that the web bundle builds. Device rendering and network connectivity still require manual testing.

Dependency audit currently reports unresolved Expo/React Native transitive advisories. A compatible dependency upgrade and another audit are required before production distribution. Do not apply `npm audit fix --force` blindly: its proposed Expo downgrade can break this app.

## Demo

Enter a MovieLens user ID (1–610), compare both Top-10 lists, and tap a movie to inspect its genres and training-history KG evidence. Global held-out metrics appear above each list. Scores are ranking scores, not calibrated probabilities.
