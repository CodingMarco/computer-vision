<template>
  <div class="w-full text-slate-300 pt-5 pl-5">
    <!-- Fix for bg-slate-900 included in body being optimized out -->
    <div class="h-0 w-0 bg-slate-900"></div>

    <h1 class="text-3xl font-bold">Computer Vision</h1>
    <RouterView />
  </div>
</template>
