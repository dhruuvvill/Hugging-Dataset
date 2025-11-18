# Fix "next: command not found" Error

## Problem
```
sh: next: command not found
```

This means dependencies aren't installed.

## Solution

### Step 1: Install Dependencies
```bash
cd frontend
npm install
```

This will install:
- next
- react
- react-dom
- tailwindcss
- autoprefixer
- postcss

### Step 2: Wait for Installation
You'll see:
```
added X packages
```

### Step 3: Start Dev Server
```bash
npm run dev
```

### Step 4: Open Browser
Go to: http://localhost:3000

## Alternative: Quick Install

If npm install is slow, try:
```bash
cd frontend
npm install --legacy-peer-deps
```

## Still Having Issues?

**Check if node_modules exists:**
```bash
ls -la frontend/node_modules
```

If it's empty, run `npm install` again.

**Check package.json:**
```bash
cat frontend/package.json
```

Make sure dependencies are listed correctly.

## Expected Output

After `npm install`, you should see:
```
+ next@15.1.5
+ react@19.x.x
+ react-dom@19.x.x
+ ...
packages installed
```

Then `npm run dev` should work!

