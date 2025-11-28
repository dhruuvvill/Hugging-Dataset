'use client';

import { motion } from 'framer-motion';

export default function GlassyBackground() {
  const shapes = [
    {
      id: 1,
      initialX: '10%',
      initialY: '20%',
      size: 400,
      gradient: 'from-purple-400/30 via-pink-400/30 to-blue-400/30',
      duration: 20,
      delay: 0,
    },
    {
      id: 2,
      initialX: '80%',
      initialY: '60%',
      size: 350,
      gradient: 'from-blue-400/30 via-cyan-400/30 to-teal-400/30',
      duration: 25,
      delay: 2,
    },
    {
      id: 3,
      initialX: '50%',
      initialY: '80%',
      size: 300,
      gradient: 'from-pink-400/30 via-rose-400/30 to-orange-400/30',
      duration: 30,
      delay: 4,
    },
    {
      id: 4,
      initialX: '20%',
      initialY: '70%',
      size: 250,
      gradient: 'from-indigo-400/30 via-purple-400/30 to-pink-400/30',
      duration: 22,
      delay: 1,
    },
    {
      id: 5,
      initialX: '70%',
      initialY: '10%',
      size: 320,
      gradient: 'from-cyan-400/30 via-blue-400/30 to-indigo-400/30',
      duration: 28,
      delay: 3,
    },
  ];

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none -z-10">
      {/* Base gradient background */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50" />
      
      {/* Animated glassy shapes */}
      {shapes.map((shape) => (
        <motion.div
          key={shape.id}
          className={`absolute rounded-full blur-3xl bg-gradient-to-br ${shape.gradient}`}
          style={{
            width: shape.size,
            height: shape.size,
            left: shape.initialX,
            top: shape.initialY,
            transform: 'translate(-50%, -50%)',
          }}
          animate={{
            x: [
              '0%',
              `${Math.random() * 40 - 20}%`,
              `${Math.random() * 40 - 20}%`,
              '0%',
            ],
            y: [
              '0%',
              `${Math.random() * 40 - 20}%`,
              `${Math.random() * 40 - 20}%`,
              '0%',
            ],
            scale: [1, 1.2, 0.8, 1],
          }}
          transition={{
            duration: shape.duration,
            repeat: Infinity,
            ease: 'easeInOut',
            delay: shape.delay,
          }}
        />
      ))}

      {/* Additional smaller floating orbs */}
      {[...Array(8)].map((_, i) => (
        <motion.div
          key={`orb-${i}`}
          className="absolute rounded-full blur-2xl bg-gradient-to-br from-white/20 to-transparent"
          style={{
            width: 100 + Math.random() * 150,
            height: 100 + Math.random() * 150,
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
          }}
          animate={{
            x: [
              '0px',
              `${Math.random() * 200 - 100}px`,
              `${Math.random() * 200 - 100}px`,
              '0px',
            ],
            y: [
              '0px',
              `${Math.random() * 200 - 100}px`,
              `${Math.random() * 200 - 100}px`,
              '0px',
            ],
            opacity: [0.3, 0.6, 0.4, 0.3],
          }}
          transition={{
            duration: 15 + Math.random() * 10,
            repeat: Infinity,
            ease: 'easeInOut',
            delay: Math.random() * 5,
          }}
        />
      ))}

      {/* Glassmorphism overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-white/10 to-transparent" />
    </div>
  );
}



