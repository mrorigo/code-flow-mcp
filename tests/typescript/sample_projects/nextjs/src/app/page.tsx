import { Suspense } from 'react';
import { HeroSection } from '@/components/HeroSection';
import { FeaturesSection } from '@/components/FeaturesSection';
import { DashboardPreview } from '@/components/DashboardPreview';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';

export default function HomePage() {
  return (
    <div className="flex flex-col min-h-screen">
      <HeroSection />

      <FeaturesSection />

      <Suspense fallback={<LoadingSpinner />}>
        <DashboardPreview />
      </Suspense>
    </div>
  );
}